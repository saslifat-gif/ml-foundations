import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertTokenizer

sys.path.insert(0, ".")
from paralla_decoder import BertEncoder, ParallelDecoder, build_dataloaders

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ── config ────────────────────────────────────────────────────────────────────
RESUME = False   # ← set True to continue from checkpoint, False to train from scratch
PROMPT_LEN = 16
COND_DROP_PROB = 0.15
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using: {device}")


class FlowNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=2048, depth=8):
        super().__init__()
        layers = [nn.Linear(latent_dim * 2 + 1, hidden_dim), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, t, z_cond):
        inp = torch.cat([z_t, z_cond, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)


def prompt_condition(z_data, attention_mask, prompt_len=PROMPT_LEN):
    prompt_z = z_data[:, :prompt_len, :]
    prompt_mask = attention_mask[:, :prompt_len].to(prompt_z.dtype).unsqueeze(-1)
    denom = prompt_mask.sum(dim=1).clamp_min(1.0)
    return (prompt_z * prompt_mask).sum(dim=1) / denom


def flow_matching_loss(flow_net, z_target, z_cond, target_mask=None):
    B, T, D = z_target.shape
    z_flat = z_target.reshape(B * T, D)
    cond_flat = z_cond.unsqueeze(1).expand(-1, T, -1).reshape(B * T, D)
    z_noise = torch.randn_like(z_flat)
    t = torch.rand(B * T, device=z_target.device)
    z_t = (1 - t.unsqueeze(-1)) * z_noise + t.unsqueeze(-1) * z_flat
    v_true = z_flat - z_noise
    v_pred = flow_net(z_t, t, cond_flat)
    loss = F.mse_loss(v_pred, v_true, reduction="none").mean(dim=-1)
    if target_mask is None:
        return loss.mean()
    mask = target_mask.reshape(B * T).to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def generate_suffix(flow_net, z_cond, batch_size, suffix_len, latent_dim, device, steps=50):
    z_cond_gen = z_cond.unsqueeze(1).expand(-1, suffix_len, -1).reshape(batch_size * suffix_len, latent_dim)
    z_gen = torch.randn(batch_size * suffix_len, latent_dim, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = torch.full((batch_size * suffix_len,), i / steps, device=device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            v = flow_net(z_gen, t_val, z_cond_gen)
        z_gen = z_gen + v * dt
    return z_gen.view(batch_size, suffix_len, latent_dim)


def evaluate(flow_net, encoder, decoder, tokenizer, val_loader, device, n_samples=4):
    flow_net.eval()
    encoder.eval()
    decoder.eval()

    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                z_data   = decoder.compress(encoder(input_ids, attention_mask))
                z_cond   = prompt_condition(z_data, attention_mask)
                z_target = z_data[:, PROMPT_LEN:, :]
                target_mask = attention_mask[:, PROMPT_LEN:]
                val_loss += flow_matching_loss(flow_net, z_target, z_cond, target_mask).item()
    avg_val_loss = val_loss / len(val_loader)

    with torch.no_grad():
        batch       = next(iter(val_loader))
        input_ids   = batch["input_ids"].to(device)
        attn_mask   = batch["attention_mask"].to(device)
        z_real      = decoder.compress(encoder(input_ids, attn_mask))
        B, S, D     = z_real.shape
        z_real_suffix = z_real[:, PROMPT_LEN:, :]
        z_real_flat = z_real_suffix.reshape(B * (S - PROMPT_LEN), D)

        z_cond = prompt_condition(z_real, attn_mask)
        z_gen_suffix = generate_suffix(flow_net, z_cond, B, S - PROMPT_LEN, D, device, steps=50)
        z_gen_flat = z_gen_suffix.reshape(B * (S - PROMPT_LEN), D)

        real_mean  = z_real_flat.mean().item()
        real_std   = z_real_flat.std().item()
        gen_mean   = z_gen_flat.mean().item()
        gen_std    = z_gen_flat.std().item()
        cosine_sim = F.cosine_similarity(
            z_real_flat.mean(0, keepdim=True),
            z_gen_flat.mean(0, keepdim=True)
        ).item()

    with torch.no_grad():
        z_gen_seq = torch.cat([z_real[:, :PROMPT_LEN, :], z_gen_suffix], dim=1)[:n_samples]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = decoder.decode_from_latent(z_gen_seq)
        pred_ids = logits.argmax(-1)
        print("\n── conditional samples ───────────────────────────────────────")
        for i in range(n_samples):
            prompt    = tokenizer.decode(input_ids[i, :PROMPT_LEN], skip_special_tokens=True)
            generated = tokenizer.decode(pred_ids[i], skip_special_tokens=True)
            print(f"  prompt:    {prompt}")
            print(f"  generated: {generated[:120]}")
            print()

    print(f"── val metrics ───────────────────────────────────────────────")
    print(f"  val flow loss : {avg_val_loss:.4f}")
    print(f"  real latents  : mean={real_mean:.3f}  std={real_std:.3f}")
    print(f"  gen  latents  : mean={gen_mean:.3f}  std={gen_std:.3f}")
    print(f"  cosine sim    : {cosine_sim:.4f}  (1.0=perfect, 0.0=orthogonal)")
    print()

    return avg_val_loss


# ── load stage 1 ──────────────────────────────────────────────────────────────
encoder = BertEncoder().to(device)
decoder = ParallelDecoder(latent_dim=256).to(device)

checkpoint = torch.load("stage1_best.pt", map_location=device, weights_only=False)
decoder.load_state_dict(checkpoint["decoder"])
if "encoder" in checkpoint:
    encoder.load_state_dict(checkpoint["encoder"])

for param in decoder.parameters():
    param.requires_grad = False
decoder.eval()
encoder.eval()
print("stage1 loaded | encoder+decoder frozen")

# ── data ──────────────────────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_loader, val_loader = build_dataloaders(
    tokenizer,
    train_size=1000000,
    batch_size=1300,
)

# ── models + optimizer ────────────────────────────────────────────────────────
flow_net = FlowNet(latent_dim=256, hidden_dim=2048, depth=8).to(device)

optimizer = AdamW([
    {"params": flow_net.parameters(), "lr": 1e-4},
])
scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
best_loss = float("inf")

# ── resume or fresh start ─────────────────────────────────────────────────────
if RESUME:
    ckpt2 = torch.load("stage2_conditional_best.pt", map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt2["flow_net"].items()}
    flow_net.load_state_dict(state)
    if "encoder" in ckpt2:
        encoder.load_state_dict(ckpt2["encoder"])
    if "best_loss" in ckpt2:
        best_loss = ckpt2["best_loss"]
    print(f"resumed from stage2_conditional_best.pt | best_loss={best_loss:.4f}")
else:
    print("training from scratch")

flow_net = torch.compile(flow_net)

EPOCHS = 20

# ── training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    flow_net.train()
    encoder.eval()
    train_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            with torch.no_grad():
                z_data = decoder.compress(encoder(input_ids, attention_mask))

            z_cond = prompt_condition(z_data, attention_mask)
            drop_mask = torch.rand(z_data.size(0), device=device) < COND_DROP_PROB
            z_cond = z_cond.masked_fill(drop_mask.unsqueeze(-1), 0.0)
            z_target = z_data[:, PROMPT_LEN:, :]
            target_mask = attention_mask[:, PROMPT_LEN:]
            loss = flow_matching_loss(flow_net, z_target, z_cond, target_mask)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(flow_net.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        if step % 50 == 0:
            print(f"epoch {epoch+1} step {step}/{len(train_loader)} | loss {loss.item():.4f}",
                  flush=True)

    avg_loss = train_loss / len(train_loader)
    print(f"\nepoch {epoch+1} done | avg train loss {avg_loss:.4f}", flush=True)

    avg_val_loss = evaluate(flow_net, encoder, decoder, tokenizer, val_loader, device)

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save({
            "flow_net": flow_net.state_dict(),
            "encoder":  encoder.state_dict(),
            "best_loss": best_loss,
        }, "stage2_conditional_best.pt")
        print(f"saved best model at val loss {best_loss:.4f}\n", flush=True)
