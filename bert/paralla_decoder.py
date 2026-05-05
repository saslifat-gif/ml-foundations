import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, BertTokenizer
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using: {device}")


# ── Models ────────────────────────────────────────────────────────────────────

class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
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
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.is_decoder = False
        self.bert      = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.to_logits = nn.Linear(768, vocab_size)

    def forward(self, z, residual_weight=1.0):
        # z: [B, seq_len, 768]
        h   = self.compress(z)                             # [B, seq_len, latent_dim]
        x   = self.project_up(h) + residual_weight * z    # annealed residual
        out = self.bert(inputs_embeds=x)
        return self.to_logits(out.last_hidden_state)       # [B, seq_len, vocab_size]

    def decode_from_latent(self, z_latent):
        """stage 2 inference: z_latent [B, seq, 256] → logits, no residual"""
        x   = self.project_up(z_latent)
        out = self.bert(inputs_embeds=x)
        return self.to_logits(out.last_hidden_state)


# ── Data ──────────────────────────────────────────────────────────────────────

def build_dataloaders(tokenizer, train_size=1000000, batch_size=128):
    ds          = load_dataset("wikitext", "wikitext-103-raw-v1")
    small_train = ds["train"].select(range(train_size))
    small_val   = ds["validation"]

    small_train = small_train.filter(lambda x: len(x["text"].strip()) > 10)
    small_val   = small_val.filter(lambda x: len(x["text"].strip()) > 10)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")

    train_tok = small_train.map(tokenize, batched=True)
    val_tok   = small_val.map(tokenize,   batched=True)
    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_tok.set_format(type="torch",   columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_tok,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")
    return train_loader, val_loader


# ── Training ──────────────────────────────────────────────────────────────────

def train(encoder, decoder, train_loader, val_loader, device, epochs=10, lr=1e-4):
    optimizer     = AdamW(decoder.parameters(), lr=lr)
    scaler        = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    VOCAB_SIZE    = 30522
    best_val_loss = float("inf")

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
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                z      = encoder(input_ids, attention_mask)
                logits = decoder(z, residual_weight=residual_weight)
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
                print(f"epoch {epoch+1} step {step}/{len(train_loader)} | loss {loss.item():.4f} | residual_weight {residual_weight:.2f}",
                      flush=True)

        avg_train = train_loss / len(train_loader)

        # ── val ───────────────────────────────────────────────────────────────
        decoder.eval()
        val_loss = 0
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

        avg_val = val_loss / len(val_loader)
        print(f"\nepoch {epoch+1} done | train {avg_train:.4f} | val {avg_val:.4f}\n", flush=True)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "decoder": decoder.state_dict(),
                "epoch":   epoch + 1,
                "val_loss": best_val_loss,
            }, "stage1_best.pt")
            print(f"saved best model at val loss {best_val_loss:.4f}", flush=True)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(text, encoder, decoder, tokenizer, max_length=128):
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

    original  = tokenizer.decode(input_ids[0],      skip_special_tokens=True)
    predicted = tokenizer.decode(pred_ids[0].cpu(),  skip_special_tokens=True)
    return original, predicted


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder   = BertEncoder().to(device)
    decoder   = ParallelDecoder(latent_dim=256).to(device)

    train_loader, val_loader = build_dataloaders(tokenizer, train_size=1000000)
    train(encoder, decoder, train_loader, val_loader, device, epochs=1)

    original, predicted = predict("the cat sat on the mat", encoder, decoder, tokenizer)
    print(f"original:  {original}")
    print(f"predicted: {predicted}")
