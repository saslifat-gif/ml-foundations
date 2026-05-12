import os
import random
import sys

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertTokenizer

sys.path.insert(0, ".")
from parallel_decoder import BertEncoder, ParallelDecoder, cached_from_pretrained
from stage2_config import (
    DECODE_LOSS_BATCH,
    EPOCHS,
    LOG_EVERY,
    MAX_SEQ_LEN,
    PROMPT_LEN,
    SEED,
    TARGET_LATENT_MEAN,
    TARGET_LATENT_STD,
    START_TRANSFORMER_HIDDEN_DIM,
    TRAIN_BATCH_SIZE,
    TRAIN_SIZE,
)
from stage2_data import build_stage2_dataloaders
from stage2_losses import rollout_cosine_alignment_loss, rollout_flow_token_ce_loss
from stage2_riemannian import DenoisingPrior, suffix_positions

# ── denoising-prior config ────────────────────────────────────────────────
DENOISING_LR         = 3e-5
DENOISING_LAYERS     = 4
DENOISING_HEADS      = 8
DENOISING_HIDDEN_DIM = START_TRANSFORMER_HIDDEN_DIM  # 512
DENOISING_MSE_WEIGHT = 1.0
DENOISING_COS_WEIGHT = 0.5
DENOISING_CE_WEIGHT  = 0.1
DENOISING_CE_BATCH   = 64
CE_WARMUP_EPOCHS     = 1    # epoch 0 trains geometry only; CE turns on from epoch 1

# Alpha curriculum — training samples only from current-phase alphas;
# val always reports all three so progress is visible across phases.
PHASE1_ALPHAS = [0.7]
PHASE2_ALPHAS = [0.7, 0.5]
PHASE3_ALPHAS = [0.7, 0.5, 0.3]
PHASE1_PROBS  = [1.0]
PHASE2_PROBS  = [0.7, 0.3]
PHASE3_PROBS  = [0.5, 0.3, 0.2]
PHASE2_EPOCH  = 0    # start here immediately on this resume
PHASE3_EPOCH  = 10
VAL_ALPHAS    = [0.3, 0.5, 0.7]

CHECKPOINT_PATH = "denoising_prior_best.pt"
RESUME = True

# ── chained inference-matched training ───────────────────────────────────
# Trains prior on the actual sampling path: pure_noise → prior(0.3) → prior(0.5) → prior(0.7)
# This matches inference exactly — no oracle z_real at any step.
CHAIN_ALPHAS = [0.3, 0.5, 0.7]   # low→high, matches inference chain order
CHAIN_LOSS_WEIGHT = 1.0           # weight relative to oracle loss
CHAIN_BATCH = 128                 # sub-batch cap (3× forward passes)
CHAIN_EPOCH = 0                   # start chain training from this epoch
# ─────────────────────────────────────────────────────────────────────────

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"denoising-prior config | "
    f"layers={DENOISING_LAYERS} heads={DENOISING_HEADS} hidden={DENOISING_HIDDEN_DIM} "
    f"lr={DENOISING_LR} | "
    f"p1={PHASE1_ALPHAS}{PHASE1_PROBS} p2={PHASE2_ALPHAS}{PHASE2_PROBS}(ep{PHASE2_EPOCH}) "
    f"p3={PHASE3_ALPHAS}{PHASE3_PROBS}(ep{PHASE3_EPOCH}) | "
    f"mse={DENOISING_MSE_WEIGHT} cos={DENOISING_COS_WEIGHT} ce={DENOISING_CE_WEIGHT} | "
    f"chain={CHAIN_ALPHAS} w={CHAIN_LOSS_WEIGHT} batch={CHAIN_BATCH} start_ep={CHAIN_EPOCH}",
    flush=True,
)

encoder = BertEncoder().to(device)
decoder = ParallelDecoder(latent_dim=256).to(device)

ckpt1 = torch.load("stage1_best.pt", map_location=device, weights_only=False)
decoder.load_state_dict(ckpt1["decoder"])
if "encoder" in ckpt1:
    encoder.load_state_dict(ckpt1["encoder"])

for p in encoder.parameters():
    p.requires_grad = False
for p in decoder.parameters():
    p.requires_grad = False
encoder.eval()
decoder.eval()
print("stage1 loaded | encoder + decoder frozen", flush=True)

tokenizer = cached_from_pretrained(BertTokenizer)
train_loader, val_loader = build_stage2_dataloaders(
    tokenizer,
    train_size=TRAIN_SIZE,
    batch_size=TRAIN_BATCH_SIZE,
    max_length=MAX_SEQ_LEN,
)

model = DenoisingPrior(
    latent_dim=256,
    hidden_dim=DENOISING_HIDDEN_DIM,
    num_layers=DENOISING_LAYERS,
    num_heads=DENOISING_HEADS,
).to(device)

optimizer = AdamW(model.parameters(), lr=DENOISING_LR)
scaler = torch.amp.GradScaler("cuda")
best_val_score = float("inf")

if RESUME and os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["denoising_prior"])
    loaded_score = ckpt.get("best_val_score", float("inf"))
    # If chain training starts at epoch 0 and checkpoint has no chain metric,
    # reset save baseline so the model saves under the new (harder) chain criterion.
    if CHAIN_EPOCH == 0 and "val_chain_sce" not in ckpt:
        best_val_score = float("inf")
        print(
            f"resumed from {CHECKPOINT_PATH} | loaded_oracle_best={loaded_score:.4f}"
            f" | reset best_val_score=inf (chain criterion active from ep0)",
            flush=True,
        )
    else:
        best_val_score = loaded_score
        print(f"resumed from {CHECKPOINT_PATH} | best_val_score={best_val_score:.4f}", flush=True)
else:
    print("training denoising prior from scratch", flush=True)

print(f"DenoisingPrior params: {sum(p.numel() for p in model.parameters()):,}", flush=True)


def _noisy_latent(z_real, alpha_val, mask=None):
    beta = (1.0 - alpha_val ** 2) ** 0.5
    z_t = alpha_val * z_real + beta * torch.randn_like(z_real)
    if mask is not None:
        z_t = z_t * mask.to(z_t.dtype).unsqueeze(-1)
    return z_t


def _prior_step(z_real, z_prompt, target_mask, suffix_ids, alpha_val, n_ce, ce_weight):
    B, T, _ = z_real.shape
    pos = suffix_positions(B, T, z_real.device, z_real.dtype)
    alpha_t = z_real.new_full((B,), alpha_val)

    z_t = _noisy_latent(z_real, alpha_val, target_mask)
    pred = model(z_t, z_prompt, alpha_t, pos, target_mask)

    if target_mask is not None and target_mask.bool().any():
        valid = target_mask.bool()
        mse = F.mse_loss(pred[valid], z_real[valid].detach())
    else:
        mse = F.mse_loss(pred, z_real.detach())

    cos_loss, cos_val = rollout_cosine_alignment_loss(pred, z_real, target_mask)

    sce = z_real.new_tensor(0.0)
    sce_p = 0.0
    if ce_weight > 0 and suffix_ids is not None:
        n = min(n_ce, B)
        z_seq = torch.cat([z_prompt[:n], pred[:n]], dim=1)
        logits = decoder.decode_from_latent(z_seq)
        sce, sce_p, _ = rollout_flow_token_ce_loss(
            logits,
            suffix_ids[:n],
            target_mask[:n] if target_mask is not None else None,
        )

    loss = DENOISING_MSE_WEIGHT * mse + DENOISING_COS_WEIGHT * cos_loss + ce_weight * sce
    return loss, mse.detach().item(), cos_val, sce.detach().item(), sce_p


def _val_alpha(z_real, z_prompt, target_mask, suffix_ids, alpha_val):
    """Oracle z_t from real target — measures denoising quality at this alpha."""
    B, T, _ = z_real.shape
    pos = suffix_positions(B, T, z_real.device, z_real.dtype)
    alpha_t = z_real.new_full((B,), alpha_val)
    z_t = _noisy_latent(z_real, alpha_val, target_mask)
    pred = model(z_t, z_prompt, alpha_t, pos, target_mask)
    n = min(DECODE_LOSS_BATCH, B)
    z_seq = torch.cat([z_prompt[:n], pred[:n]], dim=1)
    logits = decoder.decode_from_latent(z_seq)
    sce, sce_p, _ = rollout_flow_token_ce_loss(
        logits, suffix_ids[:n],
        target_mask[:n] if target_mask is not None else None,
    )
    _, cos = rollout_cosine_alignment_loss(pred, z_real, target_mask)
    return sce.item(), sce_p, cos


def _val_sampled(z_prompt, target_mask, suffix_ids, alpha_val=0.5):
    """Pure inference — no z_real; start from scaled Gaussian at alpha noise level."""
    B, T = z_prompt.size(0), MAX_SEQ_LEN - PROMPT_LEN
    pos = suffix_positions(B, T, z_prompt.device, z_prompt.dtype)
    alpha_t = z_prompt.new_full((B,), alpha_val)
    beta = (1.0 - alpha_val ** 2) ** 0.5
    z_noise = beta * TARGET_LATENT_STD * torch.randn(B, T, 256, device=z_prompt.device)
    z_noise = z_noise + TARGET_LATENT_MEAN
    if target_mask is not None:
        z_noise = z_noise * target_mask.to(z_noise.dtype).unsqueeze(-1)
    pred = model(z_noise, z_prompt, alpha_t, pos, target_mask)
    n = min(DECODE_LOSS_BATCH, B)
    z_seq = torch.cat([z_prompt[:n], pred[:n]], dim=1)
    logits = decoder.decode_from_latent(z_seq)
    sce, sce_p, _ = rollout_flow_token_ce_loss(
        logits, suffix_ids[:n],
        target_mask[:n] if target_mask is not None else None,
    )
    return sce.item(), sce_p


def _prior_chain_step(z_real, z_prompt, target_mask, suffix_ids, ce_weight):
    """
    Train prior under the actual inference distribution.
    Start from pure Gaussian; chain: prior(0.3) → prior(0.5) → prior(0.7).
    No oracle z_real at any step — gradients flow through the full chain.
    """
    B, T, D = z_real.shape
    pos = suffix_positions(B, T, z_real.device, z_real.dtype)

    z = torch.randn(B, T, D, device=z_real.device, dtype=z_real.dtype)
    z = z * TARGET_LATENT_STD + TARGET_LATENT_MEAN
    if target_mask is not None:
        z = z * target_mask.to(z.dtype).unsqueeze(-1)

    for alpha_val in CHAIN_ALPHAS:
        alpha_t = z_real.new_full((B,), alpha_val)
        z = model(z, z_prompt, alpha_t, pos, target_mask)

    if target_mask is not None and target_mask.bool().any():
        valid = target_mask.bool()
        mse = F.mse_loss(z[valid], z_real[valid].detach())
    else:
        mse = F.mse_loss(z, z_real.detach())

    cos_loss, cos_val = rollout_cosine_alignment_loss(z, z_real, target_mask)

    sce = z_real.new_tensor(0.0)
    sce_p = 0.0
    if ce_weight > 0 and suffix_ids is not None:
        n = min(DENOISING_CE_BATCH, B)
        z_seq = torch.cat([z_prompt[:n], z[:n]], dim=1)
        logits = decoder.decode_from_latent(z_seq)
        sce, sce_p, _ = rollout_flow_token_ce_loss(
            logits, suffix_ids[:n],
            target_mask[:n] if target_mask is not None else None,
        )

    loss = DENOISING_MSE_WEIGHT * mse + DENOISING_COS_WEIGHT * cos_loss + ce_weight * sce
    return loss, mse.detach().item(), cos_val, sce.detach().item(), sce_p


def _val_chain(z_prompt, target_mask, suffix_ids):
    """Chain inference val — pure noise → prior(0.3) → prior(0.5) → prior(0.7).
    This is the true inference metric: no oracle z_real at any step."""
    B, T = z_prompt.size(0), MAX_SEQ_LEN - PROMPT_LEN
    pos = suffix_positions(B, T, z_prompt.device, z_prompt.dtype)

    z = torch.randn(B, T, 256, device=z_prompt.device, dtype=z_prompt.dtype)
    z = z * TARGET_LATENT_STD + TARGET_LATENT_MEAN
    if target_mask is not None:
        z = z * target_mask.to(z.dtype).unsqueeze(-1)

    for alpha_val in CHAIN_ALPHAS:
        alpha_t = z_prompt.new_full((B,), alpha_val)
        z = model(z, z_prompt, alpha_t, pos, target_mask)

    n = min(DECODE_LOSS_BATCH, B)
    z_seq = torch.cat([z_prompt[:n], z[:n]], dim=1)
    logits = decoder.decode_from_latent(z_seq)
    sce, sce_p, _ = rollout_flow_token_ce_loss(
        logits, suffix_ids[:n],
        target_mask[:n] if target_mask is not None else None,
    )
    return sce.item(), sce_p


for epoch in range(EPOCHS):
    # ── phase selection ───────────────────────────────────────────────────
    if epoch >= PHASE3_EPOCH:
        train_alphas, train_probs, phase = PHASE3_ALPHAS, PHASE3_PROBS, 3
    elif epoch >= PHASE2_EPOCH:
        train_alphas, train_probs, phase = PHASE2_ALPHAS, PHASE2_PROBS, 2
    else:
        train_alphas, train_probs, phase = PHASE1_ALPHAS, PHASE1_PROBS, 1

    ce_weight = DENOISING_CE_WEIGHT if epoch >= CE_WARMUP_EPOCHS else 0.0

    model.train()
    train_loss = 0.0
    train_steps = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        alpha_val = random.choices(train_alphas, weights=train_probs, k=1)[0]

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            with torch.no_grad():
                z_data = decoder.compress(encoder(input_ids, attention_mask))
            z_prompt = z_data[:, :PROMPT_LEN, :]
            z_real = z_data[:, PROMPT_LEN:, :]
            target_mask = attention_mask[:, PROMPT_LEN:]
            suffix_ids = input_ids[:, PROMPT_LEN:]

            loss, mse, cos, sce, sce_p = _prior_step(
                z_real, z_prompt, target_mask, suffix_ids, alpha_val, DENOISING_CE_BATCH, ce_weight
            )

            chain_mse = chain_cos = chain_sce = chain_p = 0.0
            if epoch >= CHAIN_EPOCH and CHAIN_LOSS_WEIGHT > 0:
                n_ch = min(CHAIN_BATCH, z_real.size(0))
                chain_loss, chain_mse, chain_cos, chain_sce, chain_p = _prior_chain_step(
                    z_real[:n_ch], z_prompt[:n_ch],
                    target_mask[:n_ch] if target_mask is not None else None,
                    suffix_ids[:n_ch],
                    ce_weight,
                )
                loss = loss + CHAIN_LOSS_WEIGHT * chain_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_steps += 1

        if step % LOG_EVERY == 0:
            print(
                f"ep{epoch+1}[p{phase}] step {step}/{len(train_loader)} a={alpha_val:.1f}"
                f" | loss {loss.item():.4f}"
                f" | smse {mse:.4f} cos {cos:.3f} sce {sce:.4f}*{ce_weight:.2f} p={sce_p:.3f}"
                f" | chain mse {chain_mse:.4f} cos {chain_cos:.3f} sce {chain_sce:.4f} p={chain_p:.3f}",
                flush=True,
            )

    avg_train = train_loss / max(train_steps, 1)
    print(f"\nep{epoch+1}[p{phase}] done | avg train loss {avg_train:.4f}", flush=True)

    # ── validation — always report all VAL_ALPHAS ─────────────────────────
    model.eval()
    val_acc = {a: {"sce": 0.0, "p": 0.0, "cos": 0.0, "n": 0} for a in VAL_ALPHAS}
    val_sampled_acc = {"sce": 0.0, "p": 0.0, "n": 0}
    val_chain_acc   = {"sce": 0.0, "p": 0.0, "n": 0}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                z_data = decoder.compress(encoder(input_ids, attention_mask))
            z_prompt = z_data[:, :PROMPT_LEN, :]
            z_real = z_data[:, PROMPT_LEN:, :]
            target_mask = attention_mask[:, PROMPT_LEN:]
            suffix_ids = input_ids[:, PROMPT_LEN:]

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                for a in VAL_ALPHAS:
                    sce, p, cos = _val_alpha(z_real, z_prompt, target_mask, suffix_ids, a)
                    val_acc[a]["sce"] += sce
                    val_acc[a]["p"]   += p
                    val_acc[a]["cos"] += cos
                    val_acc[a]["n"]   += 1

                sce_s, p_s = _val_sampled(z_prompt, target_mask, suffix_ids, alpha_val=0.5)
                val_sampled_acc["sce"] += sce_s
                val_sampled_acc["p"]   += p_s
                val_sampled_acc["n"]   += 1

                sce_c, p_c = _val_chain(z_prompt, target_mask, suffix_ids)
                val_chain_acc["sce"] += sce_c
                val_chain_acc["p"]   += p_c
                val_chain_acc["n"]   += 1

    lines = []
    phase_score = 0.0
    for a in VAL_ALPHAS:
        n = max(val_acc[a]["n"], 1)
        avg_sce = val_acc[a]["sce"] / n
        avg_p   = val_acc[a]["p"]   / n
        avg_cos = val_acc[a]["cos"] / n
        tag = "*" if a in train_alphas else " "
        lines.append(f"a={a:.1f}{tag} sce={avg_sce:.3f} p={avg_p:.3f} cos={avg_cos:.3f}")
        if a in train_alphas:
            phase_score += avg_sce

    phase_score /= max(len(train_alphas), 1)

    n_s = max(val_sampled_acc["n"], 1)
    samp_sce = val_sampled_acc["sce"] / n_s
    samp_p   = val_sampled_acc["p"]   / n_s
    lines.append(f"sampled(a=0.5) sce={samp_sce:.3f} p={samp_p:.3f}")

    n_c = max(val_chain_acc["n"], 1)
    chain_sce = val_chain_acc["sce"] / n_c
    chain_p   = val_chain_acc["p"]   / n_c
    lines.append(f"chain{CHAIN_ALPHAS} sce={chain_sce:.3f} p={chain_p:.3f}")

    # Save on chain CE when chain training is active; oracle phase_score otherwise.
    save_score = chain_sce if epoch >= CHAIN_EPOCH else phase_score

    print(
        f"val ep{epoch+1}[p{phase}] | " + " | ".join(lines)
        + f" | phase_score={phase_score:.4f} save_score={save_score:.4f} (best {best_val_score:.4f})",
        flush=True,
    )

    if save_score < best_val_score:
        best_val_score = save_score
        torch.save(
            {
                "denoising_prior": model.state_dict(),
                "best_val_score": best_val_score,
                "val_alpha_stats": {
                    a: {k: val_acc[a][k] / max(val_acc[a]["n"], 1) for k in ("sce", "p", "cos")}
                    for a in VAL_ALPHAS
                },
                "val_sampled_sce": samp_sce,
                "val_sampled_p": samp_p,
                "val_chain_sce": chain_sce,
                "val_chain_p": chain_p,
                "chain_alphas": CHAIN_ALPHAS,
                "epoch": epoch,
                "phase": phase,
                "train_alphas": train_alphas,
                "denoising_layers": DENOISING_LAYERS,
                "denoising_heads": DENOISING_HEADS,
                "denoising_hidden_dim": DENOISING_HIDDEN_DIM,
                "prompt_len": PROMPT_LEN,
                "max_seq_len": MAX_SEQ_LEN,
            },
            CHECKPOINT_PATH,
        )
        print(f"saved {CHECKPOINT_PATH} | best_val_score={best_val_score:.4f}", flush=True)
