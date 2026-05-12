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
    START_TRANSFORMER_HEADS,
    START_TRANSFORMER_HIDDEN_DIM,
    START_TRANSFORMER_LAYERS,
    TRAIN_BATCH_SIZE,
    TRAIN_SIZE,
)
from stage2_data import build_stage2_dataloaders
from stage2_losses import rollout_cosine_alignment_loss, rollout_flow_token_ce_loss
from stage2_riemannian import StartTransformer, suffix_positions

# ── prior-training config ─────────────────────────────────────────────────
PRIOR_LR            = 3e-5
PRIOR_MSE_WEIGHT    = 1.0
PRIOR_COSINE_WEIGHT = 0.5

# Phase 1: geometry only.  Phase 2: add token CE.
# Switch happens at the start of epoch PHASE2_EPOCH (0-indexed).
# Set PHASE2_EPOCH = EPOCHS to stay in phase 1 for the full run.
PRIOR_TOKEN_CE_P1   = 0.0
PRIOR_TOKEN_CE_P2   = 0.1
PHASE2_EPOCH        = 0

CHECKPOINT_PATH = "start_transformer_best.pt"
RESUME = True
# ─────────────────────────────────────────────────────────────────────────

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}", flush=True)
print(
    f"start-prior config | "
    f"layers={START_TRANSFORMER_LAYERS} heads={START_TRANSFORMER_HEADS} hidden={START_TRANSFORMER_HIDDEN_DIM} "
    f"lr={PRIOR_LR} | "
    f"mse={PRIOR_MSE_WEIGHT} cos={PRIOR_COSINE_WEIGHT} "
    f"ce_p1={PRIOR_TOKEN_CE_P1} ce_p2={PRIOR_TOKEN_CE_P2} phase2_epoch={PHASE2_EPOCH}",
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

start_transformer = StartTransformer(
    latent_dim=256,
    num_layers=START_TRANSFORMER_LAYERS,
    num_heads=START_TRANSFORMER_HEADS,
    ffn_dim=START_TRANSFORMER_HIDDEN_DIM,
).to(device)

optimizer = AdamW(start_transformer.parameters(), lr=PRIOR_LR)
scaler = torch.amp.GradScaler("cuda")
best_val_score = float("inf")

if RESUME and os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    start_transformer.load_state_dict(ckpt["start_transformer"])
    best_val_score = ckpt.get("best_val_score", float("inf"))
    print(f"resumed from {CHECKPOINT_PATH} | best_val_score={best_val_score:.4f}", flush=True)
else:
    print("training start-prior from scratch", flush=True)

param_count = sum(p.numel() for p in start_transformer.parameters() if p.requires_grad)
print(f"StartTransformer params: {param_count:,}", flush=True)


def _prior_loss(z_prompt, z_target, target_mask, suffix_ids, n_ce_batch, token_ce_weight):
    B, T, _ = z_target.shape
    pos = suffix_positions(B, T, z_target.device, z_target.dtype)
    z_pred = start_transformer(z_prompt, pos, target_mask)

    if target_mask is not None and target_mask.bool().any():
        valid = target_mask.bool()
        mse = F.mse_loss(z_pred[valid], z_target[valid].detach())
    else:
        mse = F.mse_loss(z_pred, z_target.detach())

    cos_loss, cos_val = rollout_cosine_alignment_loss(z_pred, z_target, target_mask)

    sce = z_target.new_tensor(0.0)
    sce_p = 0.0
    if token_ce_weight > 0 and suffix_ids is not None:
        n = min(n_ce_batch, B)
        z_seq = torch.cat([z_prompt[:n], z_pred[:n]], dim=1)
        logits = decoder.decode_from_latent(z_seq)
        sce, sce_p, _ = rollout_flow_token_ce_loss(
            logits,
            suffix_ids[:n],
            target_mask[:n] if target_mask is not None else None,
        )

    loss = (
        PRIOR_MSE_WEIGHT * mse
        + PRIOR_COSINE_WEIGHT * cos_loss
        + token_ce_weight * sce
    )
    return loss, mse.detach().item(), cos_val, sce.detach().item(), sce_p


for epoch in range(EPOCHS):
    token_ce_weight = PRIOR_TOKEN_CE_P2 if epoch >= PHASE2_EPOCH else PRIOR_TOKEN_CE_P1
    phase = 2 if epoch >= PHASE2_EPOCH else 1

    start_transformer.train()
    train_loss = 0.0
    train_steps = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            with torch.no_grad():
                z_data = decoder.compress(encoder(input_ids, attention_mask))
            z_prompt = z_data[:, :PROMPT_LEN, :]
            z_target = z_data[:, PROMPT_LEN:, :]
            target_mask = attention_mask[:, PROMPT_LEN:]
            suffix_ids = input_ids[:, PROMPT_LEN:]

            loss, mse, cos, sce, sce_p = _prior_loss(
                z_prompt, z_target, target_mask, suffix_ids, DECODE_LOSS_BATCH, token_ce_weight
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(start_transformer.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_steps += 1

        if step % LOG_EVERY == 0:
            print(
                f"ep{epoch+1}[p{phase}] step {step}/{len(train_loader)}"
                f" | loss {loss.item():.4f}"
                f" | smse {mse:.4f}*{PRIOR_MSE_WEIGHT:.1f}={PRIOR_MSE_WEIGHT*mse:.4f}"
                f" | scos cos={cos:.3f} w*loss={PRIOR_COSINE_WEIGHT*(1-cos):.4f}"
                f" | sce {sce:.4f}*{token_ce_weight:.2f}={token_ce_weight*sce:.4f} p={sce_p:.3f}",
                flush=True,
            )

    avg_train = train_loss / max(train_steps, 1)
    print(f"\nep{epoch+1}[p{phase}] done | avg train loss {avg_train:.4f}", flush=True)

    start_transformer.eval()
    val_mse_sum = val_cos_sum = val_sce_sum = val_sce_p_sum = 0.0
    val_steps = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                z_data = decoder.compress(encoder(input_ids, attention_mask))
                z_prompt = z_data[:, :PROMPT_LEN, :]
                z_target = z_data[:, PROMPT_LEN:, :]
                target_mask = attention_mask[:, PROMPT_LEN:]
                suffix_ids = input_ids[:, PROMPT_LEN:]

                _, mse, cos, sce, sce_p = _prior_loss(
                    z_prompt, z_target, target_mask, suffix_ids, DECODE_LOSS_BATCH, token_ce_weight
                )

            val_mse_sum += mse
            val_cos_sum += cos
            val_sce_sum += sce
            val_sce_p_sum += sce_p
            val_steps += 1

    val_mse = val_mse_sum / max(val_steps, 1)
    val_cos = val_cos_sum / max(val_steps, 1)
    val_sce = val_sce_sum / max(val_steps, 1)
    val_sce_p = val_sce_p_sum / max(val_steps, 1)
    val_score = (
        PRIOR_MSE_WEIGHT * val_mse
        + PRIOR_COSINE_WEIGHT * (1.0 - val_cos)
        + token_ce_weight * val_sce
    )

    print(
        f"val[p{phase}] | smse {val_mse:.4f} | scos cos={val_cos:.3f} | sce {val_sce:.4f} p={val_sce_p:.3f}"
        f" | score {val_score:.4f} (best {best_val_score:.4f})",
        flush=True,
    )

    if val_score < best_val_score:
        best_val_score = val_score
        torch.save(
            {
                "start_transformer": start_transformer.state_dict(),
                "best_val_score": best_val_score,
                "val_mse": val_mse,
                "val_cos": val_cos,
                "val_sce": val_sce,
                "val_sce_p": val_sce_p,
                "epoch": epoch,
                "phase": phase,
                "prior_lr": PRIOR_LR,
                "prior_mse_weight": PRIOR_MSE_WEIGHT,
                "prior_cosine_weight": PRIOR_COSINE_WEIGHT,
                "prior_token_ce_weight": token_ce_weight,
                "start_transformer_layers": START_TRANSFORMER_LAYERS,
                "start_transformer_heads": START_TRANSFORMER_HEADS,
                "start_transformer_hidden_dim": START_TRANSFORMER_HIDDEN_DIM,
                "prompt_len": PROMPT_LEN,
                "max_seq_len": MAX_SEQ_LEN,
            },
            CHECKPOINT_PATH,
        )
        print(f"saved {CHECKPOINT_PATH} | best_val_score={best_val_score:.4f}", flush=True)
