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
    FLOW_DEPTH,
    FLOW_HIDDEN_DIM,
    LOG_EVERY,
    MAX_SEQ_LEN,
    METRIC_HIDDEN_DIM,
    METRIC_LOG_BOUND,
    ODE_STEPS,
    PROMPT_LEN,
    SEED,
    START_TRANSFORMER_HIDDEN_DIM,
    TARGET_LATENT_MEAN,
    TARGET_LATENT_STD,
    TRAIN_BATCH_SIZE,
    TRAIN_SIZE,
    AUX_LOGIT_FUSION_BETA,
)
from stage2_data import build_stage2_dataloaders
from stage2_losses import rollout_cosine_alignment_loss, rollout_flow_token_ce_loss
from stage2_riemannian import AuxTokenHead, DenoisingPrior, FlowNet, MetricNet, suffix_positions


# DraftPrior milestone config.
DRAFT_ALPHA = 0.7
DRAFT_CURRICULUM = (
    (0, 0.00, 0.00),  # epochs 1-2: exact target draft
    (2, 0.05, 0.00),  # epochs 3-4: slightly damaged target draft
    (4, 0.05, 0.00),  # epochs 5+: hold the useful 95% real draft band
)
DRAFT_LR = 3e-5
DRAFT_LAYERS = 4
DRAFT_HEADS = 8
DRAFT_HIDDEN_DIM = START_TRANSFORMER_HIDDEN_DIM
DRAFT_CE_WEIGHT = 1.0
DRAFT_MSE_WEIGHT = 0.05
DRAFT_COS_WEIGHT = 0.05
DRAFT_NORM_WEIGHT = 0.01
DRAFT_CE_BATCH = 64
CHECKPOINT_PATH = "draft_prior_best.pt"
RESUME = False

STAGE2_EVAL_PATH = os.environ.get("DRAFT_PRIOR_STAGE2", "")
SAVE_EXAMPLES_PATH = "draft_prior_examples.txt"
EXAMPLE_COUNT = 20
EXAMPLE_EVERY_EPOCH = True


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"draft-prior config | alpha={DRAFT_ALPHA} curriculum={DRAFT_CURRICULUM} "
    f"layers={DRAFT_LAYERS} heads={DRAFT_HEADS} hidden={DRAFT_HIDDEN_DIM} lr={DRAFT_LR} | "
    f"loss ce={DRAFT_CE_WEIGHT} mse={DRAFT_MSE_WEIGHT} cos={DRAFT_COS_WEIGHT} norm={DRAFT_NORM_WEIGHT}",
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

common_words = [
    "the", "a", "of", "and", "in", "to", "is", "was", "for", "with",
    "on", "as", "by", "from", "that", "this", "it", "are", "were", "be",
]
COMMON_TOKEN_IDS = [
    tokenizer.convert_tokens_to_ids(tok)
    for tok in common_words
    if tokenizer.convert_tokens_to_ids(tok) != tokenizer.unk_token_id
]
PAD_ID = tokenizer.pad_token_id
MASK_ID = tokenizer.mask_token_id
CLS_ID = tokenizer.cls_token_id
SEP_ID = tokenizer.sep_token_id
SPECIAL_IDS = set(tokenizer.all_special_ids)
PUNCT_TOKENS = {".", ",", ";", ":", "!", "?", "-", "(", ")", "'", '"'}


model = DenoisingPrior(
    latent_dim=256,
    hidden_dim=DRAFT_HIDDEN_DIM,
    num_layers=DRAFT_LAYERS,
    num_heads=DRAFT_HEADS,
).to(device)

optimizer = AdamW(model.parameters(), lr=DRAFT_LR)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
best_val_score = float("inf")

if RESUME and os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state = ckpt.get("draft_prior", ckpt.get("denoising_prior"))
    model.load_state_dict(state)
    best_val_score = ckpt.get("best_val_score", float("inf"))
    print(f"resumed from {CHECKPOINT_PATH} | best_val_score={best_val_score:.4f}", flush=True)
else:
    print("training draft prior from scratch", flush=True)

print(f"DraftPrior params: {sum(p.numel() for p in model.parameters()):,}", flush=True)


def _is_important_token(token):
    clean = token[2:] if token.startswith("##") else token
    if clean in PUNCT_TOKENS:
        return True
    if any(ch.isdigit() for ch in clean):
        return True
    if len(clean) >= 6 and clean.isalpha():
        return True
    return False


def draft_corruption_for_epoch(epoch):
    drop_prob, replace_prob = DRAFT_CURRICULUM[0][1], DRAFT_CURRICULUM[0][2]
    for start_epoch, sched_drop, sched_replace in DRAFT_CURRICULUM:
        if epoch >= start_epoch:
            drop_prob, replace_prob = sched_drop, sched_replace
    return drop_prob, replace_prob


def make_synthetic_draft_ids(input_ids, attention_mask, drop_prob, replace_prob):
    """Light fixed-length draft corruption that preserves word order and most sentence glue."""
    draft = input_ids.clone()
    B = input_ids.size(0)
    suffix_len = input_ids.size(1) - PROMPT_LEN

    for b in range(B):
        suffix_ids = input_ids[b, PROMPT_LEN:].tolist()
        suffix_mask = attention_mask[b, PROMPT_LEN:].tolist()
        kept = []
        for tok_id, tok_mask in zip(suffix_ids, suffix_mask):
            if tok_mask == 0 or tok_id in SPECIAL_IDS:
                continue
            token = tokenizer.convert_ids_to_tokens(int(tok_id))
            important = _is_important_token(token)
            if (not important) and random.random() < drop_prob:
                continue
            if (
                (not important)
                and COMMON_TOKEN_IDS
                and replace_prob > 0
                and token not in PUNCT_TOKENS
                and random.random() < replace_prob
            ):
                kept.append(random.choice(COMMON_TOKEN_IDS))
            else:
                kept.append(int(tok_id))

        kept = kept[:suffix_len]
        noisy_suffix = kept + [PAD_ID] * (suffix_len - len(kept))
        draft[b, PROMPT_LEN:] = torch.tensor(noisy_suffix, device=input_ids.device, dtype=input_ids.dtype)

    draft_mask = (draft != PAD_ID).to(attention_mask.dtype)
    draft[:, 0] = CLS_ID
    draft_mask[:, :PROMPT_LEN] = attention_mask[:, :PROMPT_LEN]
    return draft, draft_mask


def encode_latents(input_ids, attention_mask):
    with torch.no_grad():
        z = decoder.compress(encoder(input_ids, attention_mask))
    return z


def noisy_draft_latents(z_draft, target_mask):
    beta = (1.0 - DRAFT_ALPHA ** 2) ** 0.5
    noise = torch.randn_like(z_draft) * TARGET_LATENT_STD + TARGET_LATENT_MEAN
    z_t = DRAFT_ALPHA * z_draft + beta * noise
    if target_mask is not None:
        z_t = z_t * target_mask.to(z_t.dtype).unsqueeze(-1)
    return z_t


def norm_gap_loss(pred, target, mask):
    pred_norm = pred.norm(dim=-1)
    target_norm = target.detach().norm(dim=-1)
    if mask is not None and mask.bool().any():
        valid = mask.bool()
        return F.smooth_l1_loss(pred_norm[valid], target_norm[valid])
    return F.smooth_l1_loss(pred_norm, target_norm)


def draft_prior_forward(input_ids, attention_mask, drop_prob, replace_prob):
    draft_ids, draft_mask = make_synthetic_draft_ids(input_ids, attention_mask, drop_prob, replace_prob)
    z_real_all = encode_latents(input_ids, attention_mask)
    z_draft_all = encode_latents(draft_ids, draft_mask)
    z_prompt = z_real_all[:, :PROMPT_LEN, :]
    z_real = z_real_all[:, PROMPT_LEN:, :]
    z_draft = z_draft_all[:, PROMPT_LEN:, :]
    target_mask = attention_mask[:, PROMPT_LEN:]
    suffix_ids = input_ids[:, PROMPT_LEN:]

    B, T, _ = z_real.shape
    pos = suffix_positions(B, T, z_real.device, z_real.dtype)
    alpha_t = z_real.new_full((B,), DRAFT_ALPHA)
    z_t = noisy_draft_latents(z_draft, target_mask)
    pred = model(z_t, z_prompt, alpha_t, pos, target_mask)
    return pred, z_prompt, z_real, z_draft, target_mask, suffix_ids, draft_ids


def compute_loss(pred, z_prompt, z_real, target_mask, suffix_ids):
    n = min(DRAFT_CE_BATCH, pred.size(0))
    z_seq = torch.cat([z_prompt[:n], pred[:n]], dim=1)
    logits = decoder.decode_from_latent(z_seq)
    ce, target_prob, top1 = rollout_flow_token_ce_loss(
        logits,
        suffix_ids[:n],
        target_mask[:n] if target_mask is not None else None,
    )

    if target_mask is not None and target_mask.bool().any():
        valid = target_mask.bool()
        mse = F.mse_loss(pred[valid], z_real[valid].detach())
    else:
        mse = F.mse_loss(pred, z_real.detach())
    cos_loss, cos_val = rollout_cosine_alignment_loss(pred, z_real, target_mask)
    nloss = norm_gap_loss(pred, z_real, target_mask)
    loss = (
        DRAFT_CE_WEIGHT * ce
        + DRAFT_MSE_WEIGHT * mse
        + DRAFT_COS_WEIGHT * cos_loss
        + DRAFT_NORM_WEIGHT * nloss
    )
    return loss, {
        "ce": ce.detach().item(),
        "p": target_prob,
        "top1": top1,
        "mse": mse.detach().item(),
        "cos": cos_val,
        "norm": nloss.detach().item(),
    }


def sample_token_ids(logits, temperature=0.0, top_k=50):
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    for token_id in tokenizer.all_special_ids:
        logits[..., token_id] = -float("inf")
    if top_k is not None and top_k > 0:
        kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).view(logits.shape[:-1])


def decode_suffix(z_prompt, z_suffix, temperature=0.0):
    z_seq = torch.cat([z_prompt, z_suffix], dim=1)
    logits = decoder.decode_from_latent(z_seq)
    ids = sample_token_ids(logits, temperature=temperature)
    return [tokenizer.decode(ids[i, PROMPT_LEN:], skip_special_tokens=True) for i in range(z_suffix.size(0))]


def load_stage2_flow():
    if not STAGE2_EVAL_PATH:
        return None, None, None, AUX_LOGIT_FUSION_BETA
    if not os.path.exists(STAGE2_EVAL_PATH):
        print(f"stage2 eval skipped | missing {STAGE2_EVAL_PATH}", flush=True)
        return None, None, None, AUX_LOGIT_FUSION_BETA

    ckpt = torch.load(STAGE2_EVAL_PATH, map_location=device, weights_only=False)
    flow_net = FlowNet(
        latent_dim=256,
        hidden_dim=ckpt.get("flow_hidden_dim", FLOW_HIDDEN_DIM),
        depth=ckpt.get("flow_depth", FLOW_DEPTH),
    ).to(device)
    metric_net = MetricNet(
        latent_dim=256,
        hidden_dim=ckpt.get("metric_hidden_dim", METRIC_HIDDEN_DIM),
        log_bound=ckpt.get("metric_log_bound", METRIC_LOG_BOUND),
    ).to(device)
    flow_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["flow_net"].items()}
    metric_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["metric_net"].items()}
    flow_net.load_state_dict(flow_state, strict=False)
    metric_net.load_state_dict(metric_state, strict=False)
    flow_net.eval()
    metric_net.eval()

    aux_token_head = None
    if ckpt.get("aux_token_head") is not None:
        aux_token_head = AuxTokenHead(
            hidden_dim=ckpt.get("aux_token_hidden_dim", FLOW_HIDDEN_DIM),
            vocab_size=ckpt1.get("vocab_size", 30522),
        ).to(device)
        aux_token_head.load_state_dict(ckpt["aux_token_head"], strict=False)
        aux_token_head.eval()

    fusion_beta = ckpt.get("aux_logit_fusion_beta", AUX_LOGIT_FUSION_BETA)
    print(f"loaded stage2 flow eval checkpoint: {STAGE2_EVAL_PATH}", flush=True)
    if aux_token_head is not None:
        print(f"loaded aux fusion head | beta={fusion_beta}", flush=True)
    return flow_net, metric_net, aux_token_head, fusion_beta


def apply_flow(flow_net, metric_net, z_start, z_prompt, target_mask):
    if flow_net is None or metric_net is None:
        return None
    B, T, D = z_start.shape
    pos = suffix_positions(B, T, z_start.device, z_start.dtype)
    z = z_start
    dt = 1.0 / ODE_STEPS
    pooled_cond = z_prompt.mean(dim=1).unsqueeze(1).expand(-1, T, -1)
    for i in range(ODE_STEPS):
        t = torch.full((B, T), i / ODE_STEPS, device=z.device, dtype=z.dtype)
        v = flow_net(z, t, z_prompt, pos, target_mask)
        g = metric_net(
            z.reshape(-1, D),
            t.reshape(-1),
            pooled_cond.reshape(-1, D),
            pos.reshape(-1),
        ).reshape_as(z)
        z = z + (v / g.clamp_min(1e-3)) * dt
        if target_mask is not None:
            z = z * target_mask.to(z.dtype).unsqueeze(-1)
    return z


def eval_ce(z_prompt, z_suffix, suffix_ids, target_mask):
    n = min(DECODE_LOSS_BATCH, z_suffix.size(0))
    logits = decoder.decode_from_latent(torch.cat([z_prompt[:n], z_suffix[:n]], dim=1))
    return eval_logits_ce(logits, suffix_ids[:n], target_mask[:n] if target_mask is not None else None)


def eval_logits_ce(logits, suffix_ids, target_mask):
    ce, p, top1 = rollout_flow_token_ce_loss(
        logits,
        suffix_ids,
        target_mask,
    )
    return ce.item(), p, top1


def fused_flow_logits(flow_net, aux_token_head, fusion_beta, z_prompt, z_suffix, target_mask):
    n = min(DECODE_LOSS_BATCH, z_suffix.size(0))
    z_seq = torch.cat([z_prompt[:n], z_suffix[:n]], dim=1)
    logits = decoder.decode_from_latent(z_seq).float()
    if aux_token_head is None or fusion_beta <= 0:
        return logits
    B, T, _ = z_suffix[:n].shape
    pos = suffix_positions(B, T, z_suffix.device, z_suffix.dtype)
    t = torch.ones((B, T), device=z_suffix.device, dtype=z_suffix.dtype)
    _v, hidden = flow_net(
        z_suffix[:n],
        t,
        z_prompt[:n],
        pos,
        target_mask[:n] if target_mask is not None else None,
        return_hidden=True,
    )
    aux_logits = aux_token_head(hidden).float()
    logits[:, PROMPT_LEN:, :] = logits[:, PROMPT_LEN:, :] + fusion_beta * aux_logits
    return logits


def decode_from_logits(logits):
    ids = sample_token_ids(logits, temperature=0.0)
    return [tokenizer.decode(ids[i, PROMPT_LEN:], skip_special_tokens=True) for i in range(logits.size(0))]


@torch.no_grad()
def write_examples(epoch, flow_net=None, metric_net=None, aux_token_head=None, fusion_beta=AUX_LOGIT_FUSION_BETA):
    model.eval()
    drop_prob, replace_prob = draft_corruption_for_epoch(epoch)
    rows = []
    made = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pred, z_prompt, _z_real, z_draft, target_mask, suffix_ids, draft_ids = draft_prior_forward(
                input_ids, attention_mask, drop_prob, replace_prob
            )
            z_flow = apply_flow(flow_net, metric_net, pred, z_prompt, target_mask)
            prior_texts = decode_suffix(z_prompt, pred, temperature=0.0)
            flow_texts = decode_suffix(z_prompt, z_flow, temperature=0.0) if z_flow is not None else None
            fused_texts = None
            if z_flow is not None and aux_token_head is not None:
                fused_logits = fused_flow_logits(flow_net, aux_token_head, fusion_beta, z_prompt, z_flow, target_mask)
                fused_texts = decode_from_logits(fused_logits)

        for i in range(pred.size(0)):
            prompt = tokenizer.decode(input_ids[i, :PROMPT_LEN], skip_special_tokens=True)
            draft = tokenizer.decode(draft_ids[i, PROMPT_LEN:], skip_special_tokens=True)
            prior = prior_texts[i]
            flow = flow_texts[i] if flow_texts is not None else ""
            fused = fused_texts[i] if fused_texts is not None and i < len(fused_texts) else ""
            rows.append(
                f"--- example {made + 1} epoch {epoch + 1}\n"
                f"prompt: {prompt}\n"
                f"draft: {draft}\n"
                f"prior output: {prior}\n"
                f"flow output: {flow}\n"
                f"fused output: {fused}\n"
            )
            made += 1
            if made >= EXAMPLE_COUNT:
                with open(SAVE_EXAMPLES_PATH, "w", encoding="utf-8") as f:
                    f.write("\n".join(rows))
                print(f"saved {made} examples to {SAVE_EXAMPLES_PATH}", flush=True)
                return


flow_eval, metric_eval, aux_eval, fusion_beta_eval = load_stage2_flow()


for epoch in range(EPOCHS):
    drop_prob, replace_prob = draft_corruption_for_epoch(epoch)
    model.train()
    train_loss = 0.0
    train_steps = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pred, z_prompt, z_real, _z_draft, target_mask, suffix_ids, _draft_ids = draft_prior_forward(
                input_ids, attention_mask, drop_prob, replace_prob
            )
            loss, stats = compute_loss(pred, z_prompt, z_real, target_mask, suffix_ids)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_steps += 1

        if step % LOG_EVERY == 0:
            print(
                f"ep{epoch+1} step {step}/{len(train_loader)} | loss {loss.item():.4f} "
                f"| drop={drop_prob:.2f} repl={replace_prob:.2f} "
                f"| ce {stats['ce']:.4f} p={stats['p']:.3f} top1={stats['top1']:.3f} "
                f"| mse {stats['mse']:.4f} cos {stats['cos']:.3f} norm {stats['norm']:.4f}",
                flush=True,
            )

    avg_train = train_loss / max(train_steps, 1)
    print(f"\nep{epoch+1} done | avg train loss {avg_train:.4f}", flush=True)

    model.eval()
    val = {"ce": 0.0, "p": 0.0, "top1": 0.0, "mse": 0.0, "cos": 0.0, "norm": 0.0, "n": 0}
    flow_val = {"ce": 0.0, "p": 0.0, "top1": 0.0, "n": 0}
    fused_val = {"ce": 0.0, "p": 0.0, "top1": 0.0, "n": 0}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred, z_prompt, z_real, _z_draft, target_mask, suffix_ids, _draft_ids = draft_prior_forward(
                    input_ids, attention_mask, drop_prob, replace_prob
                )
                _loss, stats = compute_loss(pred, z_prompt, z_real, target_mask, suffix_ids)

                for key in ("ce", "p", "top1", "mse", "cos", "norm"):
                    val[key] += stats[key]
                val["n"] += 1

                z_flow = apply_flow(flow_eval, metric_eval, pred, z_prompt, target_mask)
                if z_flow is not None:
                    fce, fp, ftop1 = eval_ce(z_prompt, z_flow, suffix_ids, target_mask)
                    flow_val["ce"] += fce
                    flow_val["p"] += fp
                    flow_val["top1"] += ftop1
                    flow_val["n"] += 1
                    if aux_eval is not None:
                        fused_logits = fused_flow_logits(
                            flow_eval, aux_eval, fusion_beta_eval, z_prompt, z_flow, target_mask
                        )
                        n_fused = min(DECODE_LOSS_BATCH, z_flow.size(0))
                        fuce, fup, futop1 = eval_logits_ce(
                            fused_logits,
                            suffix_ids[:n_fused],
                            target_mask[:n_fused] if target_mask is not None else None,
                        )
                        fused_val["ce"] += fuce
                        fused_val["p"] += fup
                        fused_val["top1"] += futop1
                        fused_val["n"] += 1

    n = max(val["n"], 1)
    val_ce = val["ce"] / n
    val_p = val["p"] / n
    val_top1 = val["top1"] / n
    val_mse = val["mse"] / n
    val_cos = val["cos"] / n
    val_norm = val["norm"] / n

    msg = (
        f"val ep{epoch+1} | prior ce={val_ce:.3f} p={val_p:.3f} top1={val_top1:.3f} "
        f"| drop={drop_prob:.2f} repl={replace_prob:.2f} "
        f"| mse={val_mse:.4f} cos={val_cos:.3f} norm={val_norm:.4f}"
    )
    if flow_val["n"] > 0:
        fn = max(flow_val["n"], 1)
        msg += (
            f" | flow ce={flow_val['ce']/fn:.3f} p={flow_val['p']/fn:.3f} "
            f"top1={flow_val['top1']/fn:.3f}"
        )
    if fused_val["n"] > 0:
        fun = max(fused_val["n"], 1)
        msg += (
            f" | fused ce={fused_val['ce']/fun:.3f} p={fused_val['p']/fun:.3f} "
            f"top1={fused_val['top1']/fun:.3f}"
        )
    msg += " | target prior p>0.08 ce<6.5"
    print(msg, flush=True)

    save_score = val_ce
    if save_score < best_val_score:
        best_val_score = save_score
        torch.save(
            {
                "draft_prior": model.state_dict(),
                "denoising_prior": model.state_dict(),
                "best_val_score": best_val_score,
                "val_prior_ce": val_ce,
                "val_prior_p": val_p,
                "val_prior_top1": val_top1,
                "draft_alpha": DRAFT_ALPHA,
                "draft_drop_prob": drop_prob,
                "draft_replace_prob": replace_prob,
                "draft_curriculum": DRAFT_CURRICULUM,
                "denoising_layers": DRAFT_LAYERS,
                "denoising_heads": DRAFT_HEADS,
                "denoising_hidden_dim": DRAFT_HIDDEN_DIM,
                "prompt_len": PROMPT_LEN,
                "max_seq_len": MAX_SEQ_LEN,
                "epoch": epoch,
                "type": "draft_prior",
            },
            CHECKPOINT_PATH,
        )
        print(f"saved {CHECKPOINT_PATH} | best_val_score={best_val_score:.4f}", flush=True)

    if EXAMPLE_EVERY_EPOCH:
        write_examples(epoch, flow_eval, metric_eval, aux_eval, fusion_beta_eval)
