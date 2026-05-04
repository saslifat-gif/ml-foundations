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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using: {device}")


class FlowNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=2048, depth=8):
        super().__init__()
        layers = [nn.Linear(latent_dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, t):
        inp = torch.cat([z_t, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)


def evaluate(flow_net, encoder, decoder, tokenizer, val_loader, device, n_samples=4):
    flow_net.eval()
    encoder.eval()
    decoder.eval()

    # ── 1. flow matching loss on val set ──────────────────────────────────────
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                z_data = decoder.compress(encoder(input_ids, attention_mask))
                B, S, D = z_data.shape
                z_flat  = z_data.view(B * S, D)
                z_noise = torch.randn_like(z_flat)
                t       = torch.rand(B * S, device=device)
                z_t     = (1 - t.unsqueeze(-1)) * z_noise + t.unsqueeze(-1) * z_flat
                v_true  = z_flat - z_noise
                v_pred  = flow_net(z_t, t)
                val_loss += F.mse_loss(v_pred, v_true).item()
    avg_val_loss = val_loss / len(val_loader)

    # ── 2. latent stats: real vs generated ────────────────────────────────────
    with torch.no_grad():
        batch      = next(iter(val_loader))
        input_ids  = batch["input_ids"].to(device)
        attn_mask  = batch["attention_mask"].to(device)
        z_real     = decoder.compress(encoder(input_ids, attn_mask))  # [B, 128, 256]
        B, S, D    = z_real.shape
        z_real_flat = z_real.view(B * S, D)

        # generate latents via euler integration (flow matching inference)
        z_gen = torch.randn(B * S, D, device=device)
        STEPS = 20
        for i in range(STEPS):
            t_val = torch.full((B * S,), i / STEPS, device=device)
            with torch.amp.autocast("cuda"):
                v = flow_net(z_gen, t_val)
            z_gen = z_gen + v / STEPS

        real_mean = z_real_flat.mean().item()
        real_std  = z_real_flat.std().item()
        gen_mean  = z_gen.mean().item()
        gen_std   = z_gen.std().item()

        # cosine similarity between real and generated latent means
        r_mu = z_real_flat.mean(0)
        g_mu = z_gen.mean(0)
        cosine_sim = F.cosine_similarity(r_mu.unsqueeze(0), g_mu.unsqueeze(0)).item()

    # ── 3. decode generated latents → text samples ────────────────────────────
    with torch.no_grad():
        z_gen_seq = z_gen.view(B, S, D)[:n_samples]   # [n, 128, 256]
        with torch.amp.autocast("cuda"):
            logits   = decoder.decode_from_latent(z_gen_seq)
        pred_ids = logits.argmax(-1)
        print("\n── generated samples ─────────────────────────────────────────")
        for i in range(n_samples):
            text = tokenizer.decode(pred_ids[i], skip_special_tokens=True)
            print(f"  [{i+1}] {text[:120]}")

    print(f"\n── val metrics ───────────────────────────────────────────────")
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
encoder.train()
print("stage1 loaded | decoder frozen | encoder unfrozen")


# ── data ──────────────────────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_loader, val_loader = build_dataloaders(
    tokenizer,
    train_size=1000000,
    batch_size=1300,
)

# ── models + optimizer ────────────────────────────────────────────────────────
flow_net = FlowNet(latent_dim=256, hidden_dim=2048, depth=8).to(device)
flow_net = torch.compile(flow_net)

optimizer = AdamW([
    {"params": flow_net.parameters(), "lr": 1e-4},
    {"params": encoder.parameters(), "lr": 1e-5},
])
scaler    = torch.amp.GradScaler("cuda")

EPOCHS    = 20
best_loss = float("inf")

# ── load stage 2 ──────────────────────────────────────────────────────────────
# # 1. init
# flow_net = FlowNet(latent_dim=256, hidden_dim=2048, depth=8).to(device)

# # 2. load checkpoint BEFORE compile
# ckpt2 = torch.load("stage2_best.pt", map_location=device, weights_only=False)
# state = {k.replace("_orig_mod.", ""): v for k, v in ckpt2["flow_net"].items()}
# flow_net.load_state_dict(state)
# if "encoder" in ckpt2:
#     encoder.load_state_dict(ckpt2["encoder"])
# print("resumed from stage2_best.pt")

# # 3. compile after loading
# flow_net = torch.compile(flow_net)

# ── training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    flow_net.train()
    encoder.train()
    train_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                z_data = decoder.compress(encoder(input_ids, attention_mask))

            B, S, D = z_data.shape
            z_flat  = z_data.view(B * S, D)
            z_noise = torch.randn_like(z_flat)
            t       = torch.rand(B * S, device=device)
            z_t     = (1 - t.unsqueeze(-1)) * z_noise + t.unsqueeze(-1) * z_flat
            v_true  = z_flat - z_noise
            v_pred  = flow_net(z_t, t)
            loss    = F.mse_loss(v_pred, v_true)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(flow_net.parameters()) + list(encoder.parameters()),
            max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        if step % 50 == 0:
            print(f"epoch {epoch+1} step {step}/{len(train_loader)} | loss {loss.item():.4f}",
                  flush=True)

    avg_loss = train_loss / len(train_loader)
    print(f"\nepoch {epoch+1} done | avg train loss {avg_loss:.4f}", flush=True)

    # ── evaluate every epoch ──────────────────────────────────────────────────
    avg_val_loss = evaluate(flow_net, encoder, decoder, tokenizer, val_loader, device)

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save({
            "flow_net": flow_net.state_dict(),
            "encoder":  encoder.state_dict(),
        }, "stage2_best.pt")
        print(f"saved best model at val loss {best_loss:.4f}\n", flush=True)