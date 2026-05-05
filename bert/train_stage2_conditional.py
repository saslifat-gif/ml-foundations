import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertTokenizer

sys.path.insert(0, ".")
from parallel_decoder import BertEncoder, ParallelDecoder, build_dataloaders

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ── config ────────────────────────────────────────────────────────────────────
RESUME = False   # ← set True to continue from checkpoint, False to train from scratch
PROMPT_LEN = 16
COND_DROP_PROB = 0.15
OT_EPS = 0.05
OT_ITERS = 30
METRIC_REG = 1e-4
METRIC_LOG_BOUND = 1.0
EUCLIDEAN_LOSS_WEIGHT = 0.25
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


class MetricNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, log_bound=METRIC_LOG_BOUND):
        super().__init__()
        self.log_bound = log_bound
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t, t, z_cond):
        inp = torch.cat([z_t, z_cond, t.unsqueeze(-1)], dim=-1)
        log_g = self.net(inp)
        log_g = log_g - log_g.mean(dim=-1, keepdim=True)
        log_g = log_g.clamp(-self.log_bound, self.log_bound)
        g_diag = torch.exp(log_g)
        g_diag = g_diag / g_diag.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return g_diag


def prompt_condition(z_data, attention_mask, prompt_len=PROMPT_LEN):
    prompt_z = z_data[:, :prompt_len, :]
    prompt_mask = attention_mask[:, :prompt_len].to(prompt_z.dtype).unsqueeze(-1)
    denom = prompt_mask.sum(dim=1).clamp_min(1.0)
    return (prompt_z * prompt_mask).sum(dim=1) / denom


def sinkhorn_ot_barycentric_targets(z_noise, z_target, target_mask=None, eps=OT_EPS, iters=OT_ITERS):
    B, T, _ = z_target.shape
    dtype = z_target.dtype
    device = z_target.device

    cost = torch.cdist(z_noise.float(), z_target.float(), p=2).pow(2)
    if target_mask is None:
        target_mask = torch.ones(B, T, device=device, dtype=dtype)
    valid = target_mask.to(dtype)
    cost = cost.masked_fill(valid[:, None, :] == 0, 1e4)

    log_k = -cost / eps
    log_a = torch.full((B, T), -math.log(T), device=device)
    b = valid / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_b = torch.log(b.clamp_min(1e-8))

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(iters):
        u = log_a - torch.logsumexp(log_k + v[:, None, :], dim=2)
        v = log_b - torch.logsumexp(log_k + u[:, :, None], dim=1)

    plan = torch.exp(log_k + u[:, :, None] + v[:, None, :]).to(dtype)
    return torch.bmm(plan * T, z_target)


def flow_matching_loss(flow_net, metric_net, z_target, z_cond, target_mask=None, return_stats=False):
    if target_mask is not None:
        has_target = target_mask.sum(dim=1) > 0
        if not has_target.any():
            zero = next(flow_net.parameters()).sum() + next(metric_net.parameters()).sum()
            if return_stats:
                return zero * 0.0, {"euclidean_loss": 0.0, "metric_mean": 0.0, "metric_std": 0.0}
            return zero * 0.0
        z_target = z_target[has_target]
        z_cond = z_cond[has_target]
        target_mask = target_mask[has_target]

    B, T, D = z_target.shape
    z_noise_seq = torch.randn_like(z_target)
    z_target_ot = sinkhorn_ot_barycentric_targets(z_noise_seq, z_target, target_mask)
    z_flat = z_target_ot.reshape(B * T, D)
    cond_flat = z_cond.unsqueeze(1).expand(-1, T, -1).reshape(B * T, D)
    z_noise = z_noise_seq.reshape(B * T, D)
    t = torch.rand(B * T, device=z_target.device)
    z_t = (1 - t.unsqueeze(-1)) * z_noise + t.unsqueeze(-1) * z_flat
    v_true = z_flat - z_noise
    v_pred = flow_net(z_t, t, cond_flat)
    g_diag = metric_net(z_t, t, cond_flat)
    err = (v_pred - v_true).pow(2)
    euclidean_loss = err.mean()
    metric_loss = (g_diag * err).mean(dim=-1).mean()
    metric_reg = METRIC_REG * g_diag.log().pow(2).mean()
    total_loss = metric_loss + EUCLIDEAN_LOSS_WEIGHT * euclidean_loss + metric_reg
    if return_stats:
        return total_loss, {
            "euclidean_loss": euclidean_loss.detach().item(),
            "metric_loss": metric_loss.detach().item(),
            "metric_mean": g_diag.detach().mean().item(),
            "metric_std": g_diag.detach().std().item(),
        }
    return total_loss


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


def evaluate(flow_net, metric_net, encoder, decoder, tokenizer, val_loader, device, n_samples=4):
    flow_net.eval()
    metric_net.eval()
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
                val_loss += flow_matching_loss(flow_net, metric_net, z_target, z_cond, target_mask).item()
    avg_val_loss = val_loss / len(val_loader)

    with torch.no_grad():
        batch       = next(iter(val_loader))
        input_ids   = batch["input_ids"].to(device)
        attn_mask   = batch["attention_mask"].to(device)
        z_real      = decoder.compress(encoder(input_ids, attn_mask))
        B, S, D     = z_real.shape
        suffix_mask = attn_mask[:, PROMPT_LEN:].bool()
        z_real_suffix = z_real[:, PROMPT_LEN:, :]
        z_real_flat = z_real_suffix[suffix_mask]

        z_cond = prompt_condition(z_real, attn_mask)
        z_gen_suffix = generate_suffix(flow_net, z_cond, B, S - PROMPT_LEN, D, device, steps=50)
        z_gen_flat = z_gen_suffix[suffix_mask]

        real_mean  = z_real_flat.mean().item()
        real_std   = z_real_flat.std().item()
        gen_mean   = z_gen_flat.mean().item()
        gen_std    = z_gen_flat.std().item()
        metric_cond = z_cond.unsqueeze(1).expand(-1, S - PROMPT_LEN, -1)[suffix_mask]
        metric_t = torch.full((z_gen_flat.size(0),), 0.5, device=device)
        metric_diag = metric_net(z_gen_flat, metric_t, metric_cond)
        metric_mean = metric_diag.mean().item()
        metric_std = metric_diag.std().item()
        cosine_sim = F.cosine_similarity(
            z_real_flat.mean(0, keepdim=True),
            z_gen_flat.mean(0, keepdim=True)
        ).item()

    with torch.no_grad():
        sample_idx = (attn_mask[:, PROMPT_LEN:].sum(dim=1) > 0).nonzero(as_tuple=False).flatten()
        sample_idx = sample_idx[:n_samples]
        z_gen_seq = torch.cat([z_real[:, :PROMPT_LEN, :], z_gen_suffix], dim=1)[sample_idx]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = decoder.decode_from_latent(z_gen_seq)
        pred_ids = logits.argmax(-1)
        print("\n── conditional samples ───────────────────────────────────────")
        for sample_pos, batch_idx in enumerate(sample_idx.tolist()):
            prompt = tokenizer.decode(input_ids[batch_idx, :PROMPT_LEN], skip_special_tokens=True)
            target = tokenizer.decode(input_ids[batch_idx, PROMPT_LEN:], skip_special_tokens=True)
            generated = tokenizer.decode(pred_ids[sample_pos, PROMPT_LEN:], skip_special_tokens=True)
            print(f"  prompt:     {prompt}")
            print(f"  target:     {target[:120]}")
            print(f"  generated:  {generated[:120]}")
            print()

    print(f"── val metrics ───────────────────────────────────────────────")
    print(f"  val flow loss : {avg_val_loss:.4f}")
    print(f"  real latents  : mean={real_mean:.3f}  std={real_std:.3f}")
    print(f"  gen  latents  : mean={gen_mean:.3f}  std={gen_std:.3f}")
    print(f"  metric diag   : mean={metric_mean:.3f}  std={metric_std:.3f}")
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
metric_net = MetricNet(latent_dim=256, hidden_dim=512).to(device)

optimizer = AdamW([
    {"params": flow_net.parameters(), "lr": 1e-4},
    {"params": metric_net.parameters(), "lr": 5e-5},
])
scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
best_loss = float("inf")

# ── resume or fresh start ─────────────────────────────────────────────────────
if RESUME:
    ckpt2 = torch.load("stage2_conditional_best.pt", map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt2["flow_net"].items()}
    flow_net.load_state_dict(state)
    if "metric_net" in ckpt2:
        metric_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt2["metric_net"].items()}
        metric_net.load_state_dict(metric_state)
    if "encoder" in ckpt2:
        encoder.load_state_dict(ckpt2["encoder"])
    if "best_loss" in ckpt2:
        best_loss = ckpt2["best_loss"]
    print(f"resumed from stage2_conditional_best.pt | best_loss={best_loss:.4f}")
else:
    print("training from scratch")

flow_net = torch.compile(flow_net)
metric_net = torch.compile(metric_net)

EPOCHS = 20

# ── training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    flow_net.train()
    metric_net.train()
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
            loss, stats = flow_matching_loss(
                flow_net, metric_net, z_target, z_cond, target_mask,
                return_stats=True,
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(flow_net.parameters()) + list(metric_net.parameters()),
            max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        if step % 50 == 0:
            print(
                f"epoch {epoch+1} step {step}/{len(train_loader)}"
                f" | rloss {loss.item():.4f}"
                f" | mloss {stats['metric_loss']:.4f}"
                f" | eloss {stats['euclidean_loss']:.4f}"
                f" | metric {stats['metric_mean']:.3f}±{stats['metric_std']:.3f}",
                flush=True,
            )

    avg_loss = train_loss / len(train_loader)
    print(f"\nepoch {epoch+1} done | avg train loss {avg_loss:.4f}", flush=True)

    avg_val_loss = evaluate(flow_net, metric_net, encoder, decoder, tokenizer, val_loader, device)

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save({
            "flow_net": flow_net.state_dict(),
            "metric_net": metric_net.state_dict(),
            "encoder":  encoder.state_dict(),
            "best_loss": best_loss,
            "ot_eps": OT_EPS,
            "ot_iters": OT_ITERS,
            "metric_reg": METRIC_REG,
            "metric_log_bound": METRIC_LOG_BOUND,
            "euclidean_loss_weight": EUCLIDEAN_LOSS_WEIGHT,
        }, "stage2_conditional_best.pt")
        print(f"saved best model at val loss {best_loss:.4f}\n", flush=True)
