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
MAX_SEQ_LEN = 128
BASE_NOISE_STD = 0.30
USE_OT = False
COMPILE_MODELS = False
FAST_DEBUG = False
TRAIN_SIZE = 1000000
TRAIN_BATCH_SIZE = 2048
FLOW_HIDDEN_DIM = 512
FLOW_DEPTH = 4
METRIC_HIDDEN_DIM = 256
LOG_EVERY = 50
OT_EPS = 0.05
OT_ITERS = 30
METRIC_REG = 1e-4
METRIC_LOG_BOUND = 1.0
EUCLIDEAN_LOSS_WEIGHT = 0.25
DECODE_LOSS_WEIGHT = 0.02
DECODE_LOSS_BATCH = 64
SAMPLED_DECODE_LOSS_WEIGHT = 0.0
SAMPLED_DECODE_BATCH = 8
SAMPLED_DECODE_STEPS = 8
SAMPLED_LATENT_LOSS_WEIGHT = 0.50
SAMPLED_STD_LOSS_WEIGHT = 0.25
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using: {device}")

if FAST_DEBUG:
    TRAIN_SIZE = 100000
    TRAIN_BATCH_SIZE = 512
    FLOW_HIDDEN_DIM = 256
    FLOW_DEPTH = 2
    METRIC_HIDDEN_DIM = 128
    LOG_EVERY = 5

print(
    "stage2 config | "
    f"train_size={TRAIN_SIZE} batch={TRAIN_BATCH_SIZE} "
    f"flow={FLOW_HIDDEN_DIM}x{FLOW_DEPTH} metric_hidden={METRIC_HIDDEN_DIM} "
    f"compile={COMPILE_MODELS} fast_debug={FAST_DEBUG}",
    flush=True,
)


class FlowNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=FLOW_HIDDEN_DIM, depth=FLOW_DEPTH):
        super().__init__()
        self.prompt_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prompt_pos = nn.Parameter(torch.zeros(1, PROMPT_LEN, hidden_dim))
        self.cond_proj = nn.Linear(PROMPT_LEN * hidden_dim, latent_dim)
        self.in_proj = nn.Linear(latent_dim * 2 + 2, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "conv": nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=5,
                    padding=2,
                    groups=hidden_dim,
                ),
                "mix": nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                ),
            })
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t, t, z_cond, pos, mask=None):
        squeeze = z_t.dim() == 2
        if squeeze:
            z_t = z_t.unsqueeze(1)
            t = t.unsqueeze(1)
            if z_cond.dim() == 2:
                z_cond = z_cond.unsqueeze(1)
            pos = pos.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)

        if z_cond.dim() == 3 and z_cond.size(1) == PROMPT_LEN:
            prompt_h = self.prompt_proj(z_cond) + self.prompt_pos
            cond = self.cond_proj(prompt_h.reshape(prompt_h.size(0), -1))
        elif z_cond.dim() == 3:
            cond = z_cond.mean(dim=1)
        else:
            cond = z_cond
        cond = cond.unsqueeze(1).expand(-1, z_t.size(1), -1)

        inp = torch.cat([z_t, cond, t.unsqueeze(-1), pos.unsqueeze(-1)], dim=-1)
        h = self.in_proj(inp)
        if mask is not None:
            h = h * mask.to(h.dtype).unsqueeze(-1)

        for block in self.blocks:
            residual = h
            x = block["norm"](h)
            x = block["conv"](x.transpose(1, 2)).transpose(1, 2)
            x = block["mix"](x)
            h = residual + x
            if mask is not None:
                h = h * mask.to(h.dtype).unsqueeze(-1)

        out = self.out_proj(self.out_norm(h))
        if squeeze:
            out = out.squeeze(1)
        return out


class MetricNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, log_bound=METRIC_LOG_BOUND):
        super().__init__()
        self.log_bound = log_bound
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t, t, z_cond, pos):
        inp = torch.cat([z_t, z_cond, t.unsqueeze(-1), pos.unsqueeze(-1)], dim=-1)
        log_g = self.net(inp)
        log_g = log_g - log_g.mean(dim=-1, keepdim=True)
        log_g = log_g.clamp(-self.log_bound, self.log_bound)
        g_diag = torch.exp(log_g)
        g_diag = g_diag / g_diag.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return g_diag


def prompt_condition(z_data, attention_mask, prompt_len=PROMPT_LEN):
    prompt_z = z_data[:, :prompt_len, :]
    prompt_mask = attention_mask[:, :prompt_len].to(prompt_z.dtype).unsqueeze(-1)
    return prompt_z * prompt_mask


def pool_prompt_condition(z_cond):
    return z_cond.mean(dim=1)


def suffix_positions(batch_size, suffix_len, device, dtype=torch.float32):
    pos = torch.arange(PROMPT_LEN, PROMPT_LEN + suffix_len, device=device, dtype=dtype)
    pos = pos / max(MAX_SEQ_LEN - 1, 1)
    return pos.unsqueeze(0).expand(batch_size, suffix_len)


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


def flow_matching_loss(
    flow_net,
    metric_net,
    z_target,
    z_cond,
    target_mask=None,
    decoder=None,
    z_prompt=None,
    suffix_ids=None,
    return_stats=False,
):
    if target_mask is not None:
        has_target = target_mask.sum(dim=1) > 0
        if not has_target.any():
            zero = next(flow_net.parameters()).sum() + next(metric_net.parameters()).sum()
            if return_stats:
                return zero * 0.0, {
                    "euclidean_loss": 0.0,
                    "metric_loss": 0.0,
                    "decode_loss": 0.0,
                    "sampled_decode_loss": 0.0,
                    "sampled_latent_loss": 0.0,
                    "sampled_std_loss": 0.0,
                    "metric_mean": 0.0,
                    "metric_std": 0.0,
                }
            return zero * 0.0
        z_target = z_target[has_target]
        z_cond = z_cond[has_target]
        target_mask = target_mask[has_target]
        if z_prompt is not None:
            z_prompt = z_prompt[has_target]
        if suffix_ids is not None:
            suffix_ids = suffix_ids[has_target]

    B, T, D = z_target.shape
    z_noise_seq = torch.randn_like(z_target) * BASE_NOISE_STD
    if USE_OT:
        z_target = sinkhorn_ot_barycentric_targets(z_noise_seq, z_target, target_mask)

    pos_seq = suffix_positions(B, T, z_target.device, z_target.dtype)
    t_seq = torch.rand(B, T, device=z_target.device)
    z_t_seq = (1 - t_seq.unsqueeze(-1)) * z_noise_seq + t_seq.unsqueeze(-1) * z_target
    v_true_seq = z_target - z_noise_seq
    v_pred_seq = flow_net(z_t_seq, t_seq, z_cond, pos_seq, target_mask)

    z_flat = z_target.reshape(B * T, D)
    pooled_cond = pool_prompt_condition(z_cond)
    cond_flat = pooled_cond.unsqueeze(1).expand(-1, T, -1).reshape(B * T, D)
    z_t_flat = z_t_seq.reshape(B * T, D)
    v_true_flat = v_true_seq.reshape(B * T, D)
    v_pred_flat = v_pred_seq.reshape(B * T, D)
    pos_flat = pos_seq.reshape(B * T)
    t_flat = t_seq.reshape(B * T)
    if target_mask is not None:
        valid_flat = target_mask.reshape(B * T).bool()
        z_flat = z_flat[valid_flat]
        cond_flat = cond_flat[valid_flat]
        z_t_flat = z_t_flat[valid_flat]
        v_true_flat = v_true_flat[valid_flat]
        v_pred_flat = v_pred_flat[valid_flat]
        pos_flat = pos_flat[valid_flat]
        t_flat = t_flat[valid_flat]

    g_diag = metric_net(z_t_flat, t_flat, cond_flat, pos_flat)
    err = (v_pred_flat - v_true_flat).pow(2)
    euclidean_loss = err.mean()
    metric_loss = (g_diag * err).mean(dim=-1).mean()
    metric_reg = METRIC_REG * g_diag.log().pow(2).mean()
    decode_loss = z_target.new_tensor(0.0)
    sampled_decode_loss = z_target.new_tensor(0.0)
    sampled_latent_loss = z_target.new_tensor(0.0)
    sampled_std_loss = z_target.new_tensor(0.0)
    if decoder is not None and z_prompt is not None and suffix_ids is not None and DECODE_LOSS_WEIGHT > 0:
        n_decode = min(DECODE_LOSS_BATCH, B)
        z_pred_suffix = z_t_seq[:n_decode] + (1.0 - t_seq[:n_decode].unsqueeze(-1)) * v_pred_seq[:n_decode]
        z_pred_seq = torch.cat([z_prompt[:n_decode], z_pred_suffix], dim=1)
        logits = decoder.decode_from_latent(z_pred_seq)
        suffix_logits = logits[:, PROMPT_LEN:, :].reshape(-1, logits.size(-1))
        suffix_targets = suffix_ids[:n_decode].reshape(-1)
        decode_loss = F.cross_entropy(suffix_logits, suffix_targets, ignore_index=0)

    if (
        z_prompt is not None
        and suffix_ids is not None
        and (SAMPLED_DECODE_LOSS_WEIGHT > 0 or SAMPLED_LATENT_LOSS_WEIGHT > 0)
    ):
        n_sampled = min(SAMPLED_DECODE_BATCH, B)
        z_sampled = z_noise_seq[:n_sampled]
        z_cond_sampled = z_cond[:n_sampled]
        pos_sampled = pos_seq[:n_sampled]
        dt = 1.0 / SAMPLED_DECODE_STEPS
        for i in range(SAMPLED_DECODE_STEPS):
            t_sampled = torch.full(
                (n_sampled, T),
                i / SAMPLED_DECODE_STEPS,
                device=z_target.device,
                dtype=z_target.dtype,
            )
            v_sampled = flow_net(z_sampled, t_sampled, z_cond_sampled, pos_sampled)
            z_sampled = z_sampled + v_sampled * dt

        z_sampled_seq = torch.cat([z_prompt[:n_sampled], z_sampled], dim=1)
        sampled_err = (z_sampled - z_target[:n_sampled]).pow(2)
        if target_mask is not None:
            sampled_valid = target_mask[:n_sampled].to(sampled_err.dtype).unsqueeze(-1)
            sampled_latent_loss = (sampled_err * sampled_valid).sum() / sampled_valid.sum().clamp_min(1.0) / D
            sampled_std = z_sampled[target_mask[:n_sampled].bool()].std()
            target_std = z_target[:n_sampled][target_mask[:n_sampled].bool()].std()
        else:
            sampled_latent_loss = sampled_err.mean()
            sampled_std = z_sampled.std()
            target_std = z_target[:n_sampled].std()
        sampled_std_loss = (sampled_std - target_std.detach()).pow(2)

        if decoder is not None and SAMPLED_DECODE_LOSS_WEIGHT > 0:
            sampled_logits = decoder.decode_from_latent(z_sampled_seq)
            sampled_suffix_logits = sampled_logits[:, PROMPT_LEN:, :].reshape(-1, sampled_logits.size(-1))
            sampled_suffix_targets = suffix_ids[:n_sampled].reshape(-1)
            sampled_decode_loss = F.cross_entropy(
                sampled_suffix_logits,
                sampled_suffix_targets,
                ignore_index=0,
            )

    total_loss = (
        metric_loss
        + EUCLIDEAN_LOSS_WEIGHT * euclidean_loss
        + DECODE_LOSS_WEIGHT * decode_loss
        + SAMPLED_DECODE_LOSS_WEIGHT * sampled_decode_loss
        + SAMPLED_LATENT_LOSS_WEIGHT * sampled_latent_loss
        + SAMPLED_STD_LOSS_WEIGHT * sampled_std_loss
        + metric_reg
    )
    if return_stats:
        return total_loss, {
            "euclidean_loss": euclidean_loss.detach().item(),
            "metric_loss": metric_loss.detach().item(),
            "decode_loss": decode_loss.detach().item(),
            "sampled_decode_loss": sampled_decode_loss.detach().item(),
            "sampled_latent_loss": sampled_latent_loss.detach().item(),
            "sampled_std_loss": sampled_std_loss.detach().item(),
            "metric_mean": g_diag.detach().mean().item(),
            "metric_std": g_diag.detach().std().item(),
        }
    return total_loss


def generate_suffix(flow_net, z_cond, batch_size, suffix_len, latent_dim, device, steps=50, guidance_scale=1.0):
    pos_gen = suffix_positions(batch_size, suffix_len, device)
    z_gen = torch.randn(batch_size, suffix_len, latent_dim, device=device) * BASE_NOISE_STD
    dt = 1.0 / steps
    for i in range(steps):
        t_val = torch.full((batch_size, suffix_len), i / steps, device=device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            v = flow_net(z_gen, t_val, z_cond, pos_gen)
            if guidance_scale != 1.0:
                v_uncond = flow_net(z_gen, t_val, torch.zeros_like(z_cond), pos_gen)
                v = v_uncond + guidance_scale * (v - v_uncond)
        z_gen = z_gen + v * dt
    return z_gen


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
        metric_cond = pool_prompt_condition(z_cond).unsqueeze(1).expand(-1, S - PROMPT_LEN, -1)[suffix_mask]
        metric_t = torch.full((z_gen_flat.size(0),), 0.5, device=device)
        metric_pos = suffix_positions(B, S - PROMPT_LEN, device)[suffix_mask]
        metric_diag = metric_net(z_gen_flat, metric_t, metric_cond, metric_pos)
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
            oracle_logits = decoder.decode_from_latent(z_real[sample_idx])
        pred_ids = logits.argmax(-1)
        oracle_ids = oracle_logits.argmax(-1)
        print("\n── conditional samples ───────────────────────────────────────")
        for sample_pos, batch_idx in enumerate(sample_idx.tolist()):
            prompt = tokenizer.decode(input_ids[batch_idx, :PROMPT_LEN], skip_special_tokens=True)
            target = tokenizer.decode(input_ids[batch_idx, PROMPT_LEN:], skip_special_tokens=True)
            generated = tokenizer.decode(pred_ids[sample_pos, PROMPT_LEN:], skip_special_tokens=True)
            oracle = tokenizer.decode(oracle_ids[sample_pos, PROMPT_LEN:], skip_special_tokens=True)
            print(f"  prompt:     {prompt}")
            print(f"  target:     {target[:120]}")
            print(f"  oracle:     {oracle[:120]}")
            print(f"  generated:  {generated[:120]}")
            print()

    latent_std_gap = abs(gen_std - real_std)
    val_score = avg_val_loss + latent_std_gap + max(0.0, 0.8 - cosine_sim)

    print(f"── val metrics ───────────────────────────────────────────────")
    print(f"  val flow loss : {avg_val_loss:.4f}")
    print(f"  real latents  : mean={real_mean:.3f}  std={real_std:.3f}")
    print(f"  gen  latents  : mean={gen_mean:.3f}  std={gen_std:.3f}")
    print(f"  metric diag   : mean={metric_mean:.3f}  std={metric_std:.3f}")
    print(f"  cosine sim    : {cosine_sim:.4f}  (1.0=perfect, 0.0=orthogonal)")
    print(f"  val score     : {val_score:.4f}  (lower=better; includes gen std/cosine)")
    print()

    return avg_val_loss, val_score


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
    train_size=TRAIN_SIZE,
    batch_size=TRAIN_BATCH_SIZE,
)

# ── models + optimizer ────────────────────────────────────────────────────────
flow_net = FlowNet(latent_dim=256, hidden_dim=FLOW_HIDDEN_DIM, depth=FLOW_DEPTH).to(device)
metric_net = MetricNet(latent_dim=256, hidden_dim=METRIC_HIDDEN_DIM).to(device)

optimizer = AdamW([
    {"params": flow_net.parameters(), "lr": 1e-4},
    {"params": metric_net.parameters(), "lr": 5e-5},
])
scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
best_score = float("inf")

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
        best_score = ckpt2["best_loss"]
    if "best_score" in ckpt2:
        best_score = ckpt2["best_score"]
    print(f"resumed from stage2_conditional_best.pt | best_score={best_score:.4f}")
else:
    print("training from scratch")

if COMPILE_MODELS:
    flow_net = torch.compile(flow_net)
    metric_net = torch.compile(metric_net)
    print("torch.compile enabled")
else:
    print("torch.compile disabled")

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
            z_cond = z_cond.masked_fill(drop_mask[:, None, None], 0.0)
            z_target = z_data[:, PROMPT_LEN:, :]
            target_mask = attention_mask[:, PROMPT_LEN:]
            loss, stats = flow_matching_loss(
                flow_net, metric_net, z_target, z_cond, target_mask,
                decoder=decoder,
                z_prompt=z_data[:, :PROMPT_LEN, :],
                suffix_ids=input_ids[:, PROMPT_LEN:],
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
        if step % LOG_EVERY == 0:
            print(
                f"epoch {epoch+1} step {step}/{len(train_loader)}"
                f" | rloss {loss.item():.4f}"
                f" | mloss {stats['metric_loss']:.4f}"
                f" | eloss {stats['euclidean_loss']:.4f}"
                f" | dloss {stats['decode_loss']:.4f}"
                f" | sdloss {stats['sampled_decode_loss']:.4f}"
                f" | slloss {stats['sampled_latent_loss']:.4f}"
                f" | ssloss {stats['sampled_std_loss']:.4f}"
                f" | metric {stats['metric_mean']:.3f}±{stats['metric_std']:.3f}",
                flush=True,
            )

    avg_loss = train_loss / len(train_loader)
    print(f"\nepoch {epoch+1} done | avg train loss {avg_loss:.4f}", flush=True)

    avg_val_loss, val_score = evaluate(flow_net, metric_net, encoder, decoder, tokenizer, val_loader, device)

    if val_score < best_score:
        best_score = val_score
        torch.save({
            "flow_net": flow_net.state_dict(),
            "metric_net": metric_net.state_dict(),
            "encoder":  encoder.state_dict(),
            "best_loss": avg_val_loss,
            "best_score": best_score,
            "ot_eps": OT_EPS,
            "ot_iters": OT_ITERS,
            "metric_reg": METRIC_REG,
            "metric_log_bound": METRIC_LOG_BOUND,
            "euclidean_loss_weight": EUCLIDEAN_LOSS_WEIGHT,
            "decode_loss_weight": DECODE_LOSS_WEIGHT,
            "decode_loss_batch": DECODE_LOSS_BATCH,
            "sampled_decode_loss_weight": SAMPLED_DECODE_LOSS_WEIGHT,
            "sampled_decode_batch": SAMPLED_DECODE_BATCH,
            "sampled_decode_steps": SAMPLED_DECODE_STEPS,
            "sampled_latent_loss_weight": SAMPLED_LATENT_LOSS_WEIGHT,
            "sampled_std_loss_weight": SAMPLED_STD_LOSS_WEIGHT,
            "use_ot": USE_OT,
            "max_seq_len": MAX_SEQ_LEN,
            "base_noise_std": BASE_NOISE_STD,
            "flow_hidden_dim": FLOW_HIDDEN_DIM,
            "flow_depth": FLOW_DEPTH,
            "metric_hidden_dim": METRIC_HIDDEN_DIM,
            "train_size": TRAIN_SIZE,
            "prompt_condition": "sequence_mean_projected",
        }, "stage2_conditional_best.pt")
        print(f"saved best model at val score {best_score:.4f} | flow loss {avg_val_loss:.4f}\n", flush=True)
