import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, BertTokenizer
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using: {device}")

MAX_LENGTH = 64
TRAIN_SIZE = 1000000
TRAIN_BATCH_SIZE = 128
EPOCHS = 3
DENOISE_LATENTS = True
LATENT_NOISE_STD_FRAC = 0.05
LATENT_NOISE_WARMUP_FRAC = 0.10
LATENT_NOISE_MIN_MULT = 0.25
LATENT_STD_EMA_DECAY = 0.99


def atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def cached_from_pretrained(cls, model_name="bert-base-uncased", **kwargs):
    def validate(obj):
        if cls is BertTokenizer:
            ids = obj("the cat sat on the mat", add_special_tokens=True)["input_ids"]
            expected_prefix = [101, 1996, 4937]
            if ids[:3] != expected_prefix or obj.vocab_size != 30522:
                raise ValueError(
                    "invalid bert-base-uncased tokenizer cache: "
                    f"vocab_size={obj.vocab_size} sample_ids={ids[:8]}"
                )
        if cls is BertConfig:
            if obj.vocab_size != 30522 or obj.hidden_size != 768:
                raise ValueError(
                    "invalid bert-base-uncased config cache: "
                    f"vocab_size={obj.vocab_size} hidden_size={obj.hidden_size}"
                )
        return obj

    try:
        return validate(cls.from_pretrained(model_name, local_files_only=True, **kwargs))
    except Exception as cache_exc:
        print(f"local cache miss for {model_name}; retrying online ({cache_exc})", flush=True)
        return validate(cls.from_pretrained(model_name, force_download=True, **kwargs))


# ── Models ────────────────────────────────────────────────────────────────────

class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = cached_from_pretrained(BertModel)
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state   # [B, seq_len, 768]


class ParallelDecoder(nn.Module):
    def __init__(self, latent_dim=256, vocab_size=30522):
        super().__init__()
        self.compress   = nn.Linear(768, latent_dim)
        self.project_up = nn.Linear(latent_dim, 768)
        config = cached_from_pretrained(BertConfig)
        config.is_decoder = False
        self.bert      = cached_from_pretrained(BertModel, config=config)
        self.to_logits = nn.Linear(768, vocab_size)

    def forward(self, z, residual_weight=1.0, latent_noise_std=0.0):
        # z: [B, seq_len, 768]
        h   = self.compress(z)                             # [B, seq_len, latent_dim]
        if latent_noise_std > 0:
            h = h + torch.randn_like(h) * latent_noise_std
        x   = self.project_up(h) + residual_weight * z    # annealed residual
        out = self.bert(inputs_embeds=x)
        return self.to_logits(out.last_hidden_state)       # [B, seq_len, vocab_size]

    def decode_from_latent(self, z_latent):
        """stage 2 inference: z_latent [B, seq, 256] → logits, no residual"""
        x   = self.project_up(z_latent)
        out = self.bert(inputs_embeds=x)
        return self.to_logits(out.last_hidden_state)


# ── Data ──────────────────────────────────────────────────────────────────────

def build_dataloaders(tokenizer, train_size=1000000, batch_size=128, max_length=128):
    ds          = load_dataset("wikitext", "wikitext-103-raw-v1")
    small_train = ds["train"].select(range(train_size))
    small_val   = ds["validation"]

    small_train = small_train.filter(lambda x: len(x["text"].strip()) > 10)
    small_val   = small_val.filter(lambda x: len(x["text"].strip()) > 10)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")

    train_tok = small_train.map(tokenize, batched=True)
    val_tok   = small_val.map(tokenize,   batched=True)
    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_tok.set_format(type="torch",   columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_tok,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}  max_length: {max_length}")
    return train_loader, val_loader


# ── Training ──────────────────────────────────────────────────────────────────

def train(encoder, decoder, train_loader, val_loader, device, epochs=10, lr=1e-4):
    optimizer     = AdamW(decoder.parameters(), lr=lr)
    scaler        = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    VOCAB_SIZE    = 30522
    best_val_loss = float("inf")
    latent_std_ema = None

    for epoch in range(epochs):
        # linear anneal: 1.0 → 0.0 over all epochs
        residual_weight = max(0.0, 1.0 - epoch / epochs)
        print(f"\nepoch {epoch+1} | residual_weight: {residual_weight:.2f}")

        encoder.eval()
        decoder.train()
        train_loss = 0

        # Instead of annealing over epochs, anneal within the single epoch by step.
        for step, batch in enumerate(train_loader):
            residual_weight = max(0.0, 1.0 - step / len(train_loader))  # 1.0 → 0.0 over steps
            progress = (epoch * len(train_loader) + step + 1) / max(1, epochs * len(train_loader))
            noise_warmup = min(1.0, progress / max(LATENT_NOISE_WARMUP_FRAC, 1e-6))
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                z      = encoder(input_ids, attention_mask)
                h_probe = decoder.compress(z)
                valid_latents = h_probe[attention_mask.bool()]
                batch_latent_std = valid_latents.detach().float().std().clamp_min(1e-6)
                if latent_std_ema is None:
                    latent_std_ema = batch_latent_std
                else:
                    latent_std_ema = (
                        LATENT_STD_EMA_DECAY * latent_std_ema
                        + (1.0 - LATENT_STD_EMA_DECAY) * batch_latent_std
                    )
                latent_noise_std = 0.0
                if DENOISE_LATENTS:
                    noise_mult = LATENT_NOISE_MIN_MULT + (1.0 - LATENT_NOISE_MIN_MULT) * noise_warmup
                    latent_noise_std = (LATENT_NOISE_STD_FRAC * noise_mult * latent_std_ema).detach().item()
                logits = decoder(
                    z,
                    residual_weight=residual_weight,
                    latent_noise_std=latent_noise_std,
                )
                loss   = F.cross_entropy(
                    logits.view(-1, VOCAB_SIZE),
                    input_ids.view(-1),
                    ignore_index=0,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            if step % 50 == 0:
                print(
                    f"epoch {epoch+1} step {step}/{len(train_loader)}"
                    f" | loss {loss.item():.4f}"
                    f" | residual_weight {residual_weight:.2f}"
                    f" | latent_std {latent_std_ema.item():.4f}"
                    f" | denoise_sigma {latent_noise_std:.5f}"
                    f" ({LATENT_NOISE_STD_FRAC:.3f}x)",
                    flush=True,
                )

        avg_train = train_loss / len(train_loader)

        # ── val ───────────────────────────────────────────────────────────────
        decoder.eval()
        val_loss = 0
        val_noisy_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    z      = encoder(input_ids, attention_mask)
                    logits = decoder(z, residual_weight=0.0)
                    val_loss += F.cross_entropy(
                        logits.view(-1, VOCAB_SIZE),
                        input_ids.view(-1),
                        ignore_index=0,
                    ).item()
                    noisy_sigma = (
                        (LATENT_NOISE_STD_FRAC * latent_std_ema).detach().item()
                        if DENOISE_LATENTS and latent_std_ema is not None
                        else 0.0
                    )
                    noisy_logits = decoder(z, residual_weight=0.0, latent_noise_std=noisy_sigma)
                    val_noisy_loss += F.cross_entropy(
                        noisy_logits.view(-1, VOCAB_SIZE),
                        input_ids.view(-1),
                        ignore_index=0,
                    ).item()
                    val_batches += 1

        avg_val = val_loss / len(val_loader)
        avg_noisy_val = val_noisy_loss / max(1, val_batches)
        print(
            f"\nepoch {epoch+1} done | train {avg_train:.4f}"
            f" | val {avg_val:.4f}"
            f" | val_noisy {avg_noisy_val:.4f}"
            f" | latent_std {latent_std_ema.item():.4f}\n",
            flush=True,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            atomic_torch_save({
                "decoder": decoder.state_dict(),
                "epoch":   epoch + 1,
                "val_loss": best_val_loss,
                "val_noisy_loss": avg_noisy_val,
                "denoise_latents": DENOISE_LATENTS,
                "latent_noise_std_frac": LATENT_NOISE_STD_FRAC,
                "latent_noise_warmup_frac": LATENT_NOISE_WARMUP_FRAC,
                "latent_noise_min_mult": LATENT_NOISE_MIN_MULT,
                "latent_std_ema": latent_std_ema.detach().item(),
                "max_length": MAX_LENGTH,
                "train_size": TRAIN_SIZE,
            }, "stage1_best.pt")
            print(f"saved best model at val loss {best_val_loss:.4f}", flush=True)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(text, encoder, decoder, tokenizer, max_length=MAX_LENGTH):
    device = next(encoder.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                       padding="max_length", truncation=True)
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z        = encoder(input_ids, attention_mask)
        logits   = decoder(z, residual_weight=0.0)   # no residual at inference
        pred_ids = logits.argmax(-1)

    original_ids = input_ids[0][attention_mask[0].bool()]
    pred_masked = pred_ids[0][attention_mask[0].bool()]
    print_decode_debug("predict input", input_ids[0], attention_mask[0], tokenizer)
    print_decode_debug("predict pred", pred_ids[0], attention_mask[0], tokenizer)
    original = decode_or_debug(original_ids, tokenizer)
    predicted = decode_or_debug(pred_masked.cpu(), tokenizer)
    return original, predicted


def decode_or_debug(ids, tokenizer):
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    if decoded.strip():
        return decoded
    tokens = tokenizer.convert_ids_to_tokens(ids.detach().cpu().tolist())
    return "<blank after skip_special_tokens> " + " ".join(tokens)


def print_decode_debug(label, ids, attention_mask, tokenizer):
    ids_cpu = ids.detach().cpu()
    mask_cpu = attention_mask.detach().cpu().bool()
    masked_ids = ids_cpu[mask_cpu]
    tokens = tokenizer.convert_ids_to_tokens(masked_ids.tolist())
    print(f"{label} ids: {masked_ids.tolist()}")
    print(f"{label} tokens: {tokens}")


def show_reconstruction(batch, encoder, decoder, tokenizer):
    input_ids = batch["input_ids"][:1].to(device)
    attention_mask = batch["attention_mask"][:1].to(device)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z = encoder(input_ids, attention_mask)
        logits = decoder(z, residual_weight=0.0)
        pred_ids = logits.argmax(-1)

    original_ids = input_ids[0][attention_mask[0].bool()]
    pred_masked = pred_ids[0][attention_mask[0].bool()]
    print_decode_debug("val input", input_ids[0], attention_mask[0], tokenizer)
    print_decode_debug("val pred", pred_ids[0], attention_mask[0], tokenizer)
    original = decode_or_debug(original_ids, tokenizer)
    predicted = decode_or_debug(pred_masked.cpu(), tokenizer)
    print(f"val original:  {original}")
    print(f"val predicted: {predicted}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder   = BertEncoder().to(device)
    decoder   = ParallelDecoder(latent_dim=256).to(device)

    train_loader, val_loader = build_dataloaders(
        tokenizer,
        train_size=TRAIN_SIZE,
        batch_size=TRAIN_BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    train(encoder, decoder, train_loader, val_loader, device, epochs=EPOCHS)

    best = torch.load("stage1_best.pt", map_location=device, weights_only=False)
    decoder.load_state_dict(best["decoder"])
    print(f"loaded best stage1 checkpoint | val_loss {best['val_loss']:.4f}")

    show_reconstruction(next(iter(val_loader)), encoder, decoder, tokenizer)

    original, predicted = predict("the cat sat on the mat", encoder, decoder, tokenizer, max_length=MAX_LENGTH)
    print(f"original:  {original}")
    print(f"predicted: {predicted}")
