import copy
import os
import random
import sys

import torch
from torch.optim import AdamW
from transformers import BertTokenizer

sys.path.insert(0, ".")
from parallel_decoder import BertEncoder, ParallelDecoder, cached_from_pretrained
from stage2_config import *
from stage2_data import build_stage2_dataloaders
from stage2_eval import evaluate
from stage2_losses import flow_matching_loss
from stage2_riemannian import (
    FlowNet,
    MetricNet,
    attention_gate_grad_stats,
    attention_gate_parameters,
    non_gate_flow_parameters,
    prompt_condition,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using: {device}")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def configure_decoder_adaptation(decoder):
    for param in decoder.parameters():
        param.requires_grad = False
    if not DECODER_ADAPT:
        decoder.eval()
        return []

    trainable = []
    for module_name in DECODER_ADAPT_MODULES:
        module = getattr(decoder, module_name, None)
        if module is None:
            print(f"decoder adapt warning: missing decoder.{module_name}", flush=True)
            continue
        for param in module.parameters():
            param.requires_grad = True
            trainable.append(param)
    decoder.train()
    decoder.bert.eval()
    decoder.compress.eval()
    print(
        "decoder adapt enabled | trainable="
        f"{','.join(DECODER_ADAPT_MODULES)} lr={DECODER_ADAPT_LR}",
        flush=True,
    )
    return trainable


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


seed_everything(SEED)
print(f"seed: {SEED}", flush=True)

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
    f"flow={FLOW_HIDDEN_DIM}x{FLOW_DEPTH} metric={METRIC_HIDDEN_DIM} "
    f"compile={COMPILE_MODELS} fast_debug={FAST_DEBUG}",
    flush=True,
)

encoder = BertEncoder().to(device)
decoder = ParallelDecoder(latent_dim=256).to(device)

checkpoint = torch.load("stage1_best.pt", map_location=device, weights_only=False)
decoder.load_state_dict(checkpoint["decoder"])
if "encoder" in checkpoint:
    encoder.load_state_dict(checkpoint["encoder"])

for param in decoder.parameters():
    param.requires_grad = False
teacher_decoder = None
if DECODER_ADAPT:
    teacher_decoder = copy.deepcopy(decoder).to(device)
    freeze_module(teacher_decoder)
decoder_adapt_params = configure_decoder_adaptation(decoder)
encoder.eval()
print(
    "stage1 loaded | encoder frozen | "
    f"decoder_adapt={DECODER_ADAPT}",
    flush=True,
)

tokenizer = cached_from_pretrained(BertTokenizer)
train_loader, val_loader = build_stage2_dataloaders(
    tokenizer,
    train_size=TRAIN_SIZE,
    batch_size=TRAIN_BATCH_SIZE,
    max_length=MAX_SEQ_LEN,
)

flow_net = FlowNet(latent_dim=256, hidden_dim=FLOW_HIDDEN_DIM, depth=FLOW_DEPTH).to(device)
metric_net = MetricNet(latent_dim=256, hidden_dim=METRIC_HIDDEN_DIM).to(device)

optimizer = AdamW([
    {"params": non_gate_flow_parameters(flow_net), "lr": 1e-4},
    {"params": attention_gate_parameters(flow_net), "lr": 1e-4 * GATE_LR_MULT},
    {"params": metric_net.parameters(), "lr": 5e-5},
] + (
    [{"params": decoder_adapt_params, "lr": DECODER_ADAPT_LR}]
    if decoder_adapt_params
    else []
))
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
best_score = float("inf")
checkpoint_path = (
    "stage2_conditional_decoder_adapt_best.pt"
    if DECODER_ADAPT
    else "stage2_conditional_best.pt"
)

if RESUME:
    try:
        ckpt2 = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        ckpt2 = None
        print(
            f"could not load {checkpoint_path} ({exc}) | training stage2 from scratch",
            flush=True,
        )

    if ckpt2 is not None:
        try:
            flow_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt2["flow_net"].items()}
            flow_net.load_state_dict(flow_state)
            if "metric_net" in ckpt2:
                metric_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt2["metric_net"].items()}
                metric_net.load_state_dict(metric_state)
            else:
                print("checkpoint has no metric_net | initialized Riemannian metric from scratch")
                best_score = float("inf")
            if "encoder" in ckpt2:
                encoder.load_state_dict(ckpt2["encoder"])
            if DECODER_ADAPT and "decoder" in ckpt2:
                decoder.load_state_dict(ckpt2["decoder"])
            if "best_score" in ckpt2 and "metric_net" in ckpt2:
                best_score = ckpt2["best_score"]
            elif "best_loss" in ckpt2 and "metric_net" in ckpt2:
                best_score = ckpt2["best_loss"]
            if ckpt2.get("metric_bound_fn") != "tanh":
                best_score = float("inf")
                print("checkpoint used hard metric clamp | resetting best_score for smooth-bound run")
            print(f"resumed from {checkpoint_path} | best_score={best_score:.4f}")
        except RuntimeError as exc:
            print(f"checkpoint architecture mismatch ({exc}) | training stage2 from scratch")
            best_score = float("inf")
else:
    print("training from scratch")

if COMPILE_MODELS:
    flow_net = torch.compile(flow_net)
    metric_net = torch.compile(metric_net)
    print("torch.compile enabled")
else:
    print("torch.compile disabled")

for epoch in range(EPOCHS):
    flow_net.train()
    metric_net.train()
    encoder.eval()
    if DECODER_ADAPT:
        decoder.train()
        decoder.bert.eval()
        decoder.compress.eval()
    else:
        decoder.eval()
    train_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
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
                flow_net,
                metric_net,
                z_target,
                z_cond,
                target_mask,
                decoder=decoder,
                z_prompt=z_data[:, :PROMPT_LEN, :],
                suffix_ids=input_ids[:, PROMPT_LEN:],
                teacher_decoder=teacher_decoder,
                return_stats=True,
                global_step=epoch * len(train_loader) + step,
                steps_per_epoch=len(train_loader),
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        self_gate_grad, cross_gate_grad = attention_gate_grad_stats(flow_net)
        torch.nn.utils.clip_grad_norm_(
            list(flow_net.parameters()) + list(metric_net.parameters()) + decoder_adapt_params,
            max_norm=1.0,
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
                f" | x0 {stats['x0_loss']:.4f}"
                f" | dloss {stats['decode_loss']:.4f}*{DECODE_LOSS_WEIGHT:.3f}={stats['weighted_decode_loss']:.4f}"
                f" | rollout {stats['rollout_loss']:.4f}"
                f" | entgap {stats['rollout_entropy_loss']:.4f}*{ROLLOUT_ENTROPY_LOSS_WEIGHT:.3f}={stats['weighted_rollout_entropy_loss']:.4f}"
                f" ent g/o={stats['rollout_gen_entropy']:.2f}/{stats['rollout_oracle_entropy']:.2f}"
                f" ent_mult={stats['rollout_entropy_mult']:.2f}"
                f" | rgce {stats['rollout_gated_gen_ce']:.4f}*{ROLLOUT_GATED_GEN_CE_WEIGHT:.3f}={stats['weighted_rollout_gated_gen_ce']:.4f}"
                f" act={stats['rollout_gated_gen_ce_active']:.2f}"
                f" top1={stats['rollout_gated_gen_ce_top1']:.3f}"
                f" | rtp {stats['rollout_target_prob_loss']:.4f}*{ROLLOUT_TARGET_PROB_WEIGHT:.3f}={stats['weighted_rollout_target_prob_loss']:.4f}"
                f" act={stats['rollout_target_prob_active']:.2f}"
                f" p={stats['rollout_target_prob_gen']:.3f}/{stats['rollout_target_prob_oracle']:.3f}"
                f" | rnloss {stats['rollout_norm_loss']:.4f}"
                f" | rdiv {stats['rollout_diversity_loss']:.4f}*{ROLLOUT_DIVERSITY_LOSS_WEIGHT:.3f}={stats['weighted_rollout_diversity_loss']:.4f}"
                f" | dace {stats['decoder_adapt_real_ce']:.4f}*{DECODER_ADAPT_REAL_CE_WEIGHT:.3f}={stats['weighted_decoder_adapt_real_ce']:.4f}"
                f" | dagce {stats['decoder_adapt_gen_ce']:.4f}*{DECODER_ADAPT_GEN_CE_WEIGHT:.3f}"
                f"x{stats['decoder_adapt_gen_ce_mult']:.2f}={stats['weighted_decoder_adapt_gen_ce']:.4f}"
                f" | dakl {stats['decoder_adapt_preserve_kl']:.4f}*{DECODER_ADAPT_PRESERVE_KL_WEIGHT:.3f}={stats['weighted_decoder_adapt_preserve_kl']:.4f}"
                f" | metric {stats['metric_mean']:.3f}+/-{stats['metric_std']:.3f}"
                f" [{stats['metric_min']:.3f},{stats['metric_max']:.3f}]"
                f" | mreg {stats['metric_reg']:.5f}"
                f"x{stats['metric_reg_mult']:.1f}"
                f" | gates s={stats['self_gate']:.4f} c={stats['cross_gate']:.4f}",
                f" | ggrad s={self_gate_grad:.2e} c={cross_gate_grad:.2e}",
                f" | greg {stats['gate_reg']:.5f}",
                flush=True,
            )

    avg_loss = train_loss / len(train_loader)
    print(f"\nepoch {epoch+1} done | avg train loss {avg_loss:.4f}", flush=True)

    avg_val_loss, val_score = evaluate(flow_net, metric_net, encoder, decoder, tokenizer, val_loader, device)

    if val_score < best_score:
        best_score = val_score
        atomic_torch_save({
            "flow_net": flow_net.state_dict(),
            "metric_net": metric_net.state_dict(),
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "best_loss": avg_val_loss,
            "best_score": best_score,
            "metric_loss_weight": METRIC_LOSS_WEIGHT,
            "euclidean_loss_weight": EUCLIDEAN_LOSS_WEIGHT,
            "x0_loss_weight": X0_LOSS_WEIGHT,
            "decode_loss_weight": DECODE_LOSS_WEIGHT,
            "decode_loss_batch": DECODE_LOSS_BATCH,
            "rollout_loss_weight": ROLLOUT_LOSS_WEIGHT,
            "rollout_entropy_loss_weight": ROLLOUT_ENTROPY_LOSS_WEIGHT,
            "rollout_entropy_margin": ROLLOUT_ENTROPY_MARGIN,
            "rollout_entropy_full_epochs": ROLLOUT_ENTROPY_FULL_EPOCHS,
            "rollout_entropy_decay_epochs": ROLLOUT_ENTROPY_DECAY_EPOCHS,
            "rollout_entropy_loss_target": "one_sided_oracle_entropy",
            "rollout_entropy_loss_decoder": "teacher_decoder" if DECODER_ADAPT else "decoder",
            "rollout_entropy_gen_latents": "raw_rollout",
            "rollout_gated_gen_ce_weight": ROLLOUT_GATED_GEN_CE_WEIGHT,
            "rollout_gated_gen_ce_top1_cap": ROLLOUT_GATED_GEN_CE_TOP1_CAP,
            "rollout_gated_gen_ce_entropy_margin": ROLLOUT_GATED_GEN_CE_ENTROPY_MARGIN,
            "rollout_gated_gen_ce_decoder": "teacher_decoder" if DECODER_ADAPT else "decoder",
            "rollout_target_prob_weight": ROLLOUT_TARGET_PROB_WEIGHT,
            "rollout_target_prob_margin": ROLLOUT_TARGET_PROB_MARGIN,
            "rollout_target_prob_top1_cap": ROLLOUT_TARGET_PROB_TOP1_CAP,
            "rollout_target_prob_decoder": "teacher_decoder" if DECODER_ADAPT else "decoder",
            "rollout_norm_loss_weight": ROLLOUT_NORM_LOSS_WEIGHT,
            "rollout_diversity_loss_weight": ROLLOUT_DIVERSITY_LOSS_WEIGHT,
            "rollout_diversity_max_tokens": ROLLOUT_DIVERSITY_MAX_TOKENS,
            "rollout_batch": ROLLOUT_BATCH,
            "rollout_train_steps": ROLLOUT_TRAIN_STEPS,
            "raw_norm_gap_score_weight": RAW_NORM_GAP_SCORE_WEIGHT,
            "collapse_uniq_target": COLLAPSE_UNIQ_TARGET,
            "collapse_maxfrac_target": COLLAPSE_MAXFRAC_TARGET,
            "collapse_uniq_score_weight": COLLAPSE_UNIQ_SCORE_WEIGHT,
            "collapse_maxfrac_score_weight": COLLAPSE_MAXFRAC_SCORE_WEIGHT,
            "decoder_adapt": DECODER_ADAPT,
            "decoder_adapt_lr": DECODER_ADAPT_LR,
            "decoder_adapt_real_ce_weight": DECODER_ADAPT_REAL_CE_WEIGHT,
            "decoder_adapt_gen_ce_weight": DECODER_ADAPT_GEN_CE_WEIGHT,
            "decoder_adapt_gen_ce_ramp_epochs": DECODER_ADAPT_GEN_CE_RAMP_EPOCHS,
            "decoder_adapt_preserve_kl_weight": DECODER_ADAPT_PRESERVE_KL_WEIGHT,
            "decoder_adapt_kl_temp": DECODER_ADAPT_KL_TEMP,
            "decoder_adapt_modules": DECODER_ADAPT_MODULES,
            "metric_reg": METRIC_REG,
            "metric_warmup_reg_mult": METRIC_WARMUP_REG_MULT,
            "metric_warmup_steps": METRIC_WARMUP_STEPS,
            "metric_log_bound": METRIC_LOG_BOUND,
            "metric_bound_fn": "tanh",
            "self_gate_scale": SELF_GATE_SCALE,
            "cross_gate_scale": CROSS_GATE_SCALE,
            "gate_init": GATE_INIT,
            "gate_reg_weight": GATE_REG_WEIGHT,
            "gate_lr_mult": GATE_LR_MULT,
            "max_seq_len": MAX_SEQ_LEN,
            "base_noise_std": BASE_NOISE_STD,
            "calibrate_generated_latents": CALIBRATE_GENERATED_LATENTS,
            "target_latent_mean": TARGET_LATENT_MEAN,
            "target_latent_std": TARGET_LATENT_STD,
            "flow_hidden_dim": FLOW_HIDDEN_DIM,
            "flow_depth": FLOW_DEPTH,
            "flow_out_init": "zero",
            "metric_hidden_dim": METRIC_HIDDEN_DIM,
            "ode_steps": ODE_STEPS,
            "eval_sample_temperature": EVAL_SAMPLE_TEMPERATURE,
            "eval_sample_top_k": EVAL_SAMPLE_TOP_K,
            "eval_sample_top_p": EVAL_SAMPLE_TOP_P,
            "train_size": TRAIN_SIZE,
            "seed": SEED,
            "dataloader_num_workers": DATALOADER_NUM_WORKERS,
            "prompt_condition": "riemannian_prompt_prefix",
            "stage2_arch": "riemannian_metric_flow_decoder_adapt" if DECODER_ADAPT else "riemannian_metric_flow",
        }, checkpoint_path)
        print(
            f"saved best model at val score {best_score:.4f} | "
            f"flow loss {avg_val_loss:.4f} | path {checkpoint_path}\n",
            flush=True,
        )
