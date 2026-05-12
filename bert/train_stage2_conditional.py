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
    AuxTokenHead,
    DenoisingPrior,
    DenoisingPriorSampler,
    DraftPriorSampler,
    FlowNet,
    LatentProjector,
    MetricNet,
    StartMLP,
    StartTransformer,
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
    if not DECODER_ADAPT or (LATENT_PROJECTOR and LATENT_PROJECTOR_ONLY):
        decoder.eval()
        return [], []

    head_trainable = []
    for module_name in DECODER_ADAPT_MODULES:
        module = getattr(decoder, module_name, None)
        if module is None:
            print(f"decoder adapt warning: missing decoder.{module_name}", flush=True)
            continue
        for param in module.parameters():
            param.requires_grad = True
            head_trainable.append(param)

    bert_trainable = []
    bert_layers = getattr(getattr(decoder.bert, "encoder", None), "layer", [])
    n_bert_layers = min(max(0, DECODER_ADAPT_BERT_LAST_N_LAYERS), len(bert_layers))
    if n_bert_layers > 0:
        for layer in bert_layers[-n_bert_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
                bert_trainable.append(param)

    decoder.train()
    decoder.bert.eval()
    if n_bert_layers > 0:
        for layer in bert_layers[-n_bert_layers:]:
            layer.train()
    decoder.compress.eval()
    print(
        "decoder adapt enabled | trainable="
        f"{','.join(DECODER_ADAPT_MODULES)} lr={DECODER_ADAPT_LR} "
        f"bert_last_n={n_bert_layers} bert_lr={DECODER_ADAPT_BERT_LR}",
        flush=True,
    )
    return head_trainable, bert_trainable


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def set_module_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable
    if trainable:
        module.train()
    else:
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
    f"flow={FLOW_HIDDEN_DIM}x{FLOW_DEPTH} refine_scale={FLOW_REFINE_SCALE} "
    f"target_frac={FLOW_REFINE_TARGET_FRACTION} vclamp={VELOCITY_CLAMP} "
    f"metric={METRIC_HIDDEN_DIM} metric_frozen_steps={METRIC_FROZEN_STEPS} "
    f"flow_token_ce={ROLLOUT_FLOW_TOKEN_CE_WEIGHT}x{ROLLOUT_FLOW_TOKEN_CE_BATCH} "
    f"fused_ce={FUSED_TOKEN_CE_WEIGHT}x{FUSED_TOKEN_CE_BATCH} beta={AUX_LOGIT_FUSION_BETA} "
    f"structured_start={STRUCTURED_TARGET_START} alpha={STRUCTURED_START_ALPHA} "
    f"start_mlp={START_MLP} start_transformer={START_TRANSFORMER} "
    f"denoising_prior={DENOISING_PRIOR} dp_alpha={DENOISING_PRIOR_ALPHA} dp_frozen={DENOISING_PRIOR_FROZEN} "
    f"draft_prior={DRAFT_PRIOR} draft_drop={DRAFT_PRIOR_DROP_PROB} "
    f"start_noise={START_NOISE_STD_FRAC} "
    f"aux_token={AUX_TOKEN_CE_WEIGHT} shared_block={TOKEN_SHARED_BLOCK}x{TOKEN_SHARED_BLOCK_SCALE} "
    f"token_residual={TOKEN_RESIDUAL_SCALE} "
    f"projector={LATENT_PROJECTOR} projector_only={LATENT_PROJECTOR_ONLY} reset={LATENT_PROJECTOR_RESET} "
    f"decoder_flow_joint={DECODER_FLOW_JOINT} decoder_generated_adapt_only={DECODER_GENERATED_ADAPT_ONLY} "
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
decoder_adapt_params, decoder_adapt_bert_params = configure_decoder_adaptation(decoder)
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
if DENOISING_PRIOR:
    _dp_ckpt = torch.load(DENOISING_PRIOR_PATH, map_location=device, weights_only=False)
    _dp = DenoisingPrior(
        latent_dim=256,
        hidden_dim=_dp_ckpt.get("denoising_hidden_dim", START_TRANSFORMER_HIDDEN_DIM),
        num_layers=_dp_ckpt.get("denoising_layers", START_TRANSFORMER_LAYERS),
        num_heads=_dp_ckpt.get("denoising_heads", START_TRANSFORMER_HEADS),
    ).to(device)
    _dp.load_state_dict(_dp_ckpt["denoising_prior"])
    if DENOISING_PRIOR_FROZEN:
        freeze_module(_dp)
    if DRAFT_PRIOR:
        start_mlp = DraftPriorSampler(
            _dp, latent_dim=256, alpha=DENOISING_PRIOR_ALPHA,
        ).to(device)
    else:
        start_mlp = DenoisingPriorSampler(
            _dp, latent_dim=256, alpha=DENOISING_PRIOR_ALPHA,
            use_oracle=DENOISING_PRIOR_ORACLE_ZT,
        ).to(device)
    print(
        f"{'draft' if DRAFT_PRIOR else 'denoising'} prior loaded from {DENOISING_PRIOR_PATH} | "
        f"alpha={DENOISING_PRIOR_ALPHA} frozen={DENOISING_PRIOR_FROZEN} "
        f"oracle_zt={DENOISING_PRIOR_ORACLE_ZT} draft_prior={DRAFT_PRIOR}",
        flush=True,
    )
elif START_TRANSFORMER:
    start_mlp = StartTransformer(
        latent_dim=256,
        num_layers=START_TRANSFORMER_LAYERS,
        num_heads=START_TRANSFORMER_HEADS,
        ffn_dim=START_TRANSFORMER_HIDDEN_DIM,
    ).to(device)
elif START_MLP:
    start_mlp = StartMLP(latent_dim=256, hidden_dim=START_MLP_HIDDEN_DIM).to(device)
else:
    start_mlp = None
aux_token_head = (
    AuxTokenHead(hidden_dim=AUX_TOKEN_HIDDEN_DIM, vocab_size=tokenizer.vocab_size).to(device)
    if AUX_TOKEN_CE_WEIGHT > 0
    else None
)
latent_projector = (
    LatentProjector(
        latent_dim=256,
        hidden_dim=LATENT_PROJECTOR_HIDDEN_DIM,
        depth=LATENT_PROJECTOR_DEPTH,
        residual_scale=LATENT_PROJECTOR_RES_SCALE,
    ).to(device)
    if LATENT_PROJECTOR
    else None
)

if (LATENT_PROJECTOR and LATENT_PROJECTOR_ONLY) or DECODER_GENERATED_ADAPT_ONLY:
    freeze_module(flow_net)
    freeze_module(metric_net)
    if LATENT_PROJECTOR and LATENT_PROJECTOR_ONLY:
        freeze_module(decoder)
        print("latent projector only | flow/metric/decoder frozen", flush=True)
    else:
        print("decoder generated adapt only | flow/metric frozen", flush=True)

optimizer = AdamW([
    {"params": non_gate_flow_parameters(flow_net), "lr": 1e-4},
    {"params": attention_gate_parameters(flow_net), "lr": 1e-4 * GATE_LR_MULT},
    {"params": metric_net.parameters(), "lr": 5e-5},
    *([{"params": [p for p in start_mlp.parameters() if p.requires_grad], "lr": START_TRANSFORMER_LR if START_TRANSFORMER else START_MLP_LR}] if start_mlp is not None and any(p.requires_grad for p in start_mlp.parameters()) else []),
    *([{"params": aux_token_head.parameters(), "lr": 1e-4}] if aux_token_head is not None else []),
] + (
    [{"params": decoder_adapt_params, "lr": DECODER_ADAPT_LR}]
    if decoder_adapt_params
    else []
) + (
    [{"params": decoder_adapt_bert_params, "lr": DECODER_ADAPT_BERT_LR}]
    if decoder_adapt_bert_params
    else []
) + (
    [{"params": latent_projector.parameters(), "lr": LATENT_PROJECTOR_LR}]
    if latent_projector is not None
    else []
))
projector_param_count = (
    sum(p.numel() for p in latent_projector.parameters() if p.requires_grad)
    if latent_projector is not None
    else 0
)
aux_param_count = (
    sum(p.numel() for p in aux_token_head.parameters() if p.requires_grad)
    if aux_token_head is not None
    else 0
)
start_param_count = (
    sum(p.numel() for p in start_mlp.parameters() if p.requires_grad)
    if start_mlp is not None
    else 0
)
optimizer_param_count = sum(
    p.numel()
    for group in optimizer.param_groups
    for p in group["params"]
    if p.requires_grad
)
print(
    f"optimizer params | projector_trainable={projector_param_count} "
    f"start_trainable={start_param_count} "
    f"aux_trainable={aux_param_count} "
    f"optimizer_trainable={optimizer_param_count}",
    flush=True,
)


PUNCT_TOKENS = {".", ",", ";", ":", "!", "?", "-", "(", ")", "'", '"'}
SPECIAL_IDS = set(tokenizer.all_special_ids)
PAD_ID = tokenizer.pad_token_id


def _is_draft_anchor(token):
    clean = token[2:] if token.startswith("##") else token
    return clean in PUNCT_TOKENS or any(ch.isdigit() for ch in clean) or (len(clean) >= 6 and clean.isalpha())


def make_stage2_draft_ids(input_ids, attention_mask):
    """Stage2 draft source: 95% real target, order preserved, no replacement by default."""
    if not DRAFT_PRIOR:
        return None, None
    draft = input_ids.clone()
    suffix_len = input_ids.size(1) - PROMPT_LEN
    for b in range(input_ids.size(0)):
        kept = []
        for tok_id, tok_mask in zip(input_ids[b, PROMPT_LEN:].tolist(), attention_mask[b, PROMPT_LEN:].tolist()):
            if tok_mask == 0 or tok_id in SPECIAL_IDS:
                continue
            token = tokenizer.convert_ids_to_tokens(int(tok_id))
            if (not _is_draft_anchor(token)) and random.random() < DRAFT_PRIOR_DROP_PROB:
                continue
            kept.append(int(tok_id))
        kept = kept[:suffix_len]
        draft[b, PROMPT_LEN:] = torch.tensor(
            kept + [PAD_ID] * (suffix_len - len(kept)),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
    draft_mask = (draft != PAD_ID).to(attention_mask.dtype)
    draft_mask[:, :PROMPT_LEN] = attention_mask[:, :PROMPT_LEN]
    return draft, draft_mask


def make_stage2_draft_latents(input_ids, attention_mask):
    if not DRAFT_PRIOR:
        return None
    draft_ids, draft_mask = make_stage2_draft_ids(input_ids, attention_mask)
    with torch.no_grad():
        z_draft = decoder.compress(encoder(draft_ids, draft_mask))
    return z_draft[:, PROMPT_LEN:, :]
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
best_score = float("inf")
checkpoint_path = (
    "stage2_conditional_projector_best.pt"
    if LATENT_PROJECTOR
    else
    "stage2_conditional_flow_decoder_joint_best.pt"
    if DECODER_FLOW_JOINT
    else
    "stage2_conditional_decoder_generated_adapt_best.pt"
    if DECODER_GENERATED_ADAPT_ONLY
    else
    "stage2_conditional_decoder_adapt_best.pt"
    if DECODER_ADAPT
    else "stage2_conditional_best.pt"
)

if RESUME:
    loaded_checkpoint_path = (
        "stage2_conditional_decoder_adapt_best.pt"
        if LATENT_PROJECTOR and LATENT_PROJECTOR_RESET
        else checkpoint_path
    )
    try:
        ckpt2 = torch.load(loaded_checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        fallback_path = (
            "stage2_conditional_decoder_adapt_best.pt"
            if (LATENT_PROJECTOR or DECODER_GENERATED_ADAPT_ONLY or DECODER_FLOW_JOINT)
            and loaded_checkpoint_path != "stage2_conditional_decoder_adapt_best.pt"
            else None
        )
        if fallback_path is not None and fallback_path != checkpoint_path:
            try:
                ckpt2 = torch.load(fallback_path, map_location=device, weights_only=False)
                loaded_checkpoint_path = fallback_path
                print(
                    f"could not load {checkpoint_path} ({exc}) | "
                    f"initializing projector from {fallback_path}",
                    flush=True,
                )
            except Exception as fallback_exc:
                ckpt2 = None
                print(
                    f"could not load {checkpoint_path} ({exc}) or {fallback_path} ({fallback_exc})",
                    flush=True,
                )
        else:
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
                if LATENT_PROJECTOR and LATENT_PROJECTOR_ONLY:
                    freeze_module(decoder)
                if DECODER_GENERATED_ADAPT_ONLY and teacher_decoder is not None:
                    teacher_decoder = copy.deepcopy(decoder).to(device)
                    freeze_module(teacher_decoder)
            if latent_projector is not None and "latent_projector" in ckpt2 and not LATENT_PROJECTOR_RESET:
                latent_projector.load_state_dict(ckpt2["latent_projector"])
            if start_mlp is not None and "start_mlp" in ckpt2 and ckpt2["start_mlp"] is not None:
                start_mlp.load_state_dict(ckpt2["start_mlp"])
            if aux_token_head is not None and "aux_token_head" in ckpt2 and ckpt2["aux_token_head"] is not None:
                aux_token_head.load_state_dict(ckpt2["aux_token_head"])
            if (
                LATENT_PROJECTOR
                and (LATENT_PROJECTOR_RESET or "latent_projector" not in ckpt2 or loaded_checkpoint_path != checkpoint_path)
            ):
                best_score = float("inf")
                print("initializing fresh latent_projector | resetting best_score")
            elif "best_score" in ckpt2 and "metric_net" in ckpt2:
                best_score = ckpt2["best_score"]
            elif "best_loss" in ckpt2 and "metric_net" in ckpt2:
                best_score = ckpt2["best_loss"]
            if ckpt2.get("metric_bound_fn") != "tanh":
                best_score = float("inf")
                print("checkpoint used hard metric clamp | resetting best_score for smooth-bound run")
            print(f"resumed from {loaded_checkpoint_path} | best_score={best_score:.4f}")
        except RuntimeError as exc:
            print(f"checkpoint architecture mismatch ({exc}) | training stage2 from scratch")
            best_score = float("inf")
    elif LATENT_PROJECTOR and LATENT_PROJECTOR_ONLY:
        raise RuntimeError("LATENT_PROJECTOR_ONLY requires an existing stage2 checkpoint to project from")
else:
    print("training from scratch")

if LATENT_PROJECTOR and LATENT_PROJECTOR_ONLY and not RESUME:
    raise RuntimeError("LATENT_PROJECTOR_ONLY requires RESUME=True and a trained stage2 checkpoint")

if COMPILE_MODELS:
    flow_net = torch.compile(flow_net)
    metric_net = torch.compile(metric_net)
    print("torch.compile enabled")
else:
    print("torch.compile disabled")

for epoch in range(EPOCHS):
    if (LATENT_PROJECTOR and LATENT_PROJECTOR_ONLY) or DECODER_GENERATED_ADAPT_ONLY:
        flow_net.eval()
        metric_net.eval()
        if aux_token_head is not None:
            aux_token_head.train()
        if start_mlp is not None:
            start_mlp.train()
        if latent_projector is not None:
            latent_projector.train()
    else:
        flow_net.train()
        if aux_token_head is not None:
            aux_token_head.train()
        if start_mlp is not None:
            start_mlp.train()
        if latent_projector is not None:
            latent_projector.train()
    encoder.eval()
    if LATENT_PROJECTOR and LATENT_PROJECTOR_ONLY:
        decoder.eval()
    elif DECODER_ADAPT:
        decoder.train()
        decoder.bert.eval()
        bert_layers = getattr(getattr(decoder.bert, "encoder", None), "layer", [])
        n_bert_layers = min(max(0, DECODER_ADAPT_BERT_LAST_N_LAYERS), len(bert_layers))
        if n_bert_layers > 0:
            for layer in bert_layers[-n_bert_layers:]:
                layer.train()
        decoder.compress.eval()
    else:
        decoder.eval()
    train_loss = 0

    for step, batch in enumerate(train_loader):
        global_step = epoch * len(train_loader) + step
        metric_trainable = not (
            METRIC_FROZEN_STEPS > 0
            and global_step < METRIC_FROZEN_STEPS
            and not (LATENT_PROJECTOR and LATENT_PROJECTOR_ONLY)
            and not DECODER_GENERATED_ADAPT_ONLY
        )
        set_module_trainable(metric_net, metric_trainable)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            with torch.no_grad():
                z_data = decoder.compress(encoder(input_ids, attention_mask))

            z_cond = prompt_condition(z_data, attention_mask)
            drop_mask = torch.rand(z_data.size(0), device=device) < COND_DROP_PROB
            z_cond = z_cond.masked_fill(drop_mask[:, None, None], 0.0)
            z_target = z_data[:, PROMPT_LEN:, :]
            z_draft_start = make_stage2_draft_latents(input_ids, attention_mask)
            target_mask = attention_mask[:, PROMPT_LEN:]
            loss, stats = flow_matching_loss(
                flow_net,
                metric_net,
                z_target,
                z_cond,
                target_mask,
                aux_token_head=aux_token_head,
                decoder=decoder,
                z_prompt=z_data[:, :PROMPT_LEN, :],
                suffix_ids=input_ids[:, PROMPT_LEN:],
                teacher_decoder=teacher_decoder,
                start_mlp=start_mlp,
                latent_projector=latent_projector,
                z_draft_start=z_draft_start,
                return_stats=True,
                global_step=global_step,
                steps_per_epoch=len(train_loader),
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        self_gate_grad, cross_gate_grad = attention_gate_grad_stats(flow_net)
        if latent_projector is not None:
            projector_grad_terms = [
                param.grad.detach().float().pow(2).sum()
                for param in latent_projector.parameters()
                if param.grad is not None
            ]
            projector_grad_norm = (
                torch.sqrt(torch.stack(projector_grad_terms).sum()).item()
                if projector_grad_terms
                else 0.0
            )
        else:
            projector_grad_norm = 0.0
        torch.nn.utils.clip_grad_norm_(
            (
                list(flow_net.parameters())
                + list(metric_net.parameters())
                + (list(start_mlp.parameters()) if start_mlp is not None else [])
                + (list(aux_token_head.parameters()) if aux_token_head is not None else [])
                + decoder_adapt_params
                + decoder_adapt_bert_params
                + (list(latent_projector.parameters()) if latent_projector is not None else [])
            ),
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
                f" | auxce {stats['aux_token_ce']:.4f}*{AUX_TOKEN_CE_WEIGHT:.3f}={stats['weighted_aux_token_ce']:.4f}"
                f" acc={stats['aux_token_acc']:.3f}"
                f" p={stats['aux_token_target_prob']:.3f}"
                f" | smse {stats['start_mse_loss']:.4f}*{START_MLP_MSE_WEIGHT:.2f}={stats['weighted_start_mse_loss']:.4f}"
                f" | scos {stats['start_cosine_loss']:.4f}*{START_MLP_COSINE_WEIGHT:.2f}={stats['weighted_start_cosine_loss']:.4f}"
                f" cos={stats['start_cosine']:.3f}"
                f" | sce {stats['start_token_ce']:.4f}*{START_MLP_TOKEN_CE_WEIGHT:.2f}={stats['weighted_start_token_ce']:.4f}"
                f" p={stats['start_target_prob']:.3f}"
                f" sig={stats['start_noise_std']:.4f}"
                f" a={stats['structured_start_alpha']:.2f}"
                f" | tblk {stats['token_block_norm']:.4f}/{stats['token_hidden_norm']:.4f}"
                f" r={stats['token_block_ratio']:.4f}"
                f" | vout {stats['velocity_out_norm']:.4f}"
                f" w={stats['out_proj_weight_norm']:.4f}"
                f" | rollout {stats['rollout_loss']:.4f}"
                f" | entgap {stats['rollout_entropy_loss']:.4f}*{ROLLOUT_ENTROPY_LOSS_WEIGHT:.3f}={stats['weighted_rollout_entropy_loss']:.4f}"
                f" ent g/o={stats['rollout_gen_entropy']:.2f}/{stats['rollout_oracle_entropy']:.2f}"
                f" ent_mult={stats['rollout_entropy_mult']:.2f}"
                f" | rgce {stats['rollout_gated_gen_ce']:.4f}*{ROLLOUT_GATED_GEN_CE_WEIGHT:.3f}={stats['weighted_rollout_gated_gen_ce']:.4f}"
                f" act={stats['rollout_gated_gen_ce_active']:.2f}"
                f" top1={stats['rollout_gated_gen_ce_top1']:.3f}"
                f" | rfce {stats['rollout_flow_token_ce']:.4f}*{ROLLOUT_FLOW_TOKEN_CE_WEIGHT:.3f}={stats['weighted_rollout_flow_token_ce']:.4f}"
                f" p={stats['rollout_flow_token_ce_target_prob']:.3f}"
                f" top1={stats['rollout_flow_token_ce_top1']:.3f}"
                f" | fce {stats['fused_token_ce']:.4f}*{FUSED_TOKEN_CE_WEIGHT:.3f}={stats['weighted_fused_token_ce']:.4f}"
                f" p={stats['fused_token_target_prob']:.3f}"
                f" top1={stats['fused_token_top1']:.3f}"
                f" | rtp {stats['rollout_target_prob_loss']:.4f}*{ROLLOUT_TARGET_PROB_WEIGHT:.3f}={stats['weighted_rollout_target_prob_loss']:.4f}"
                f" act={stats['rollout_target_prob_active']:.2f}"
                f" p={stats['rollout_target_prob_gen']:.3f}/{stats['rollout_target_prob_oracle']:.3f}"
                f" | rnloss {stats['rollout_norm_loss']:.4f}"
                f" | rdiv {stats['rollout_diversity_loss']:.4f}*{ROLLOUT_DIVERSITY_LOSS_WEIGHT:.3f}={stats['weighted_rollout_diversity_loss']:.4f}"
                f" | rcos {stats['rollout_cosine_loss']:.4f}*{ROLLOUT_COSINE_LOSS_WEIGHT:.3f}={stats['weighted_rollout_cosine_loss']:.4f}"
                f" cos={stats['rollout_cosine']:.3f}"
                f" | pmse {stats['projector_mse_loss']:.4f}*{LATENT_PROJECTOR_MSE_WEIGHT:.3f}={stats['weighted_projector_mse_loss']:.4f}"
                f" | pcos {stats['projector_cosine_loss']:.4f}*{LATENT_PROJECTOR_COSINE_WEIGHT:.3f}={stats['weighted_projector_cosine_loss']:.4f}"
                f" cos={stats['projector_cosine']:.3f}"
                f" | pce {stats['projector_token_ce']:.4f}*{LATENT_PROJECTOR_TOKEN_CE_WEIGHT:.3f}={stats['weighted_projector_token_ce']:.4f}"
                f" p={stats['projector_target_prob']:.3f}"
                f" | preg {stats['projector_delta_reg']:.4f}*{LATENT_PROJECTOR_DELTA_REG_WEIGHT:.3f}={stats['weighted_projector_delta_reg']:.4f}"
                f" dnorm={stats['projector_delta_norm']:.6f}"
                f" dabs={stats['projector_delta_abs_mean']:.6f}/{stats['projector_delta_abs_max']:.6f}"
                f" zstd={stats['projector_z_std']:.4f}"
                f" pgrad={projector_grad_norm:.2e}"
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

    avg_val_loss, val_score = evaluate(
        flow_net,
        metric_net,
        encoder,
        decoder,
        tokenizer,
        val_loader,
        device,
        start_mlp=start_mlp,
        aux_token_head=aux_token_head,
        latent_projector=latent_projector,
        draft_start_fn=make_stage2_draft_latents if DRAFT_PRIOR else None,
    )

    if val_score < best_score:
        best_score = val_score
        atomic_torch_save({
            "flow_net": flow_net.state_dict(),
            "metric_net": metric_net.state_dict(),
            "start_mlp": start_mlp.state_dict() if start_mlp is not None else None,
            "aux_token_head": aux_token_head.state_dict() if aux_token_head is not None else None,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "latent_projector": latent_projector.state_dict() if latent_projector is not None else None,
            "best_loss": avg_val_loss,
            "best_score": best_score,
            "metric_loss_weight": METRIC_LOSS_WEIGHT,
            "euclidean_loss_weight": EUCLIDEAN_LOSS_WEIGHT,
            "x0_loss_weight": X0_LOSS_WEIGHT,
            "decode_loss_weight": DECODE_LOSS_WEIGHT,
            "flow_refine_scale": FLOW_REFINE_SCALE,
            "flow_refine_target_fraction": FLOW_REFINE_TARGET_FRACTION,
            "velocity_clamp": VELOCITY_CLAMP,
            "metric_frozen_steps": METRIC_FROZEN_STEPS,
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
            "rollout_flow_token_ce_weight": ROLLOUT_FLOW_TOKEN_CE_WEIGHT,
            "rollout_flow_token_ce_batch": ROLLOUT_FLOW_TOKEN_CE_BATCH,
            "rollout_flow_token_ce_decoder": "teacher_decoder" if DECODER_ADAPT else "decoder",
            "fused_token_ce_weight": FUSED_TOKEN_CE_WEIGHT,
            "fused_token_ce_batch": FUSED_TOKEN_CE_BATCH,
            "aux_logit_fusion_beta": AUX_LOGIT_FUSION_BETA,
            "rollout_target_prob_weight": ROLLOUT_TARGET_PROB_WEIGHT,
            "rollout_target_prob_margin": ROLLOUT_TARGET_PROB_MARGIN,
            "rollout_target_prob_top1_cap": ROLLOUT_TARGET_PROB_TOP1_CAP,
            "rollout_target_prob_decoder": "teacher_decoder" if DECODER_ADAPT else "decoder",
            "rollout_norm_loss_weight": ROLLOUT_NORM_LOSS_WEIGHT,
            "rollout_diversity_loss_weight": ROLLOUT_DIVERSITY_LOSS_WEIGHT,
            "rollout_cosine_loss_weight": ROLLOUT_COSINE_LOSS_WEIGHT,
            "rollout_diversity_max_tokens": ROLLOUT_DIVERSITY_MAX_TOKENS,
            "rollout_batch": ROLLOUT_BATCH,
            "rollout_train_steps": ROLLOUT_TRAIN_STEPS,
            "structured_target_start": STRUCTURED_TARGET_START,
            "structured_start_alpha": STRUCTURED_START_ALPHA,
            "start_mlp": START_MLP,
            "start_mlp_hidden_dim": START_MLP_HIDDEN_DIM,
            "start_transformer": START_TRANSFORMER,
            "denoising_prior": DENOISING_PRIOR,
            "denoising_prior_path": DENOISING_PRIOR_PATH,
            "denoising_prior_alpha": DENOISING_PRIOR_ALPHA,
            "denoising_prior_frozen": DENOISING_PRIOR_FROZEN,
            "draft_prior": DRAFT_PRIOR,
            "draft_prior_path": DENOISING_PRIOR_PATH if DRAFT_PRIOR else None,
            "draft_prior_drop_prob": DRAFT_PRIOR_DROP_PROB,
            "draft_prior_replace_prob": DRAFT_PRIOR_REPLACE_PROB,
            "start_transformer_layers": START_TRANSFORMER_LAYERS,
            "start_transformer_heads": START_TRANSFORMER_HEADS,
            "start_transformer_hidden_dim": START_TRANSFORMER_HIDDEN_DIM,
            "start_transformer_lr": START_TRANSFORMER_LR,
            "start_noise_std_frac": START_NOISE_STD_FRAC,
            "start_mlp_lr": START_MLP_LR,
            "start_mlp_mse_weight": START_MLP_MSE_WEIGHT,
            "start_mlp_cosine_weight": START_MLP_COSINE_WEIGHT,
            "start_mlp_token_ce_weight": START_MLP_TOKEN_CE_WEIGHT,
            "aux_token_ce_weight": AUX_TOKEN_CE_WEIGHT,
            "aux_token_hidden_dim": AUX_TOKEN_HIDDEN_DIM,
            "aux_token_batch": AUX_TOKEN_BATCH,
            "token_shared_block": TOKEN_SHARED_BLOCK,
            "token_shared_block_scale": TOKEN_SHARED_BLOCK_SCALE,
            "token_residual_scale": TOKEN_RESIDUAL_SCALE,
            "latent_projector": LATENT_PROJECTOR,
            "latent_projector_only": LATENT_PROJECTOR_ONLY,
            "latent_projector_reset": LATENT_PROJECTOR_RESET,
            "latent_projector_hidden_dim": LATENT_PROJECTOR_HIDDEN_DIM,
            "latent_projector_depth": LATENT_PROJECTOR_DEPTH,
            "latent_projector_res_scale": LATENT_PROJECTOR_RES_SCALE,
            "latent_projector_lr": LATENT_PROJECTOR_LR,
            "latent_projector_mse_weight": LATENT_PROJECTOR_MSE_WEIGHT,
            "latent_projector_cosine_weight": LATENT_PROJECTOR_COSINE_WEIGHT,
            "latent_projector_token_ce_weight": LATENT_PROJECTOR_TOKEN_CE_WEIGHT,
            "latent_projector_delta_reg_weight": LATENT_PROJECTOR_DELTA_REG_WEIGHT,
            "raw_norm_gap_score_weight": RAW_NORM_GAP_SCORE_WEIGHT,
            "collapse_uniq_target": COLLAPSE_UNIQ_TARGET,
            "collapse_maxfrac_target": COLLAPSE_MAXFRAC_TARGET,
            "collapse_uniq_score_weight": COLLAPSE_UNIQ_SCORE_WEIGHT,
            "collapse_maxfrac_score_weight": COLLAPSE_MAXFRAC_SCORE_WEIGHT,
            "decoder_adapt": DECODER_ADAPT,
            "decoder_flow_joint": DECODER_FLOW_JOINT,
            "decoder_generated_adapt_only": DECODER_GENERATED_ADAPT_ONLY,
            "decoder_adapt_detach_generated": DECODER_ADAPT_DETACH_GENERATED,
            "decoder_adapt_lr": DECODER_ADAPT_LR,
            "decoder_adapt_bert_lr": DECODER_ADAPT_BERT_LR,
            "decoder_adapt_bert_last_n_layers": DECODER_ADAPT_BERT_LAST_N_LAYERS,
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
            "stage2_arch": (
                "riemannian_metric_flow_latent_projector"
                if LATENT_PROJECTOR
                else
                "riemannian_metric_flow_decoder_joint"
                if DECODER_FLOW_JOINT
                else
                "riemannian_metric_flow_decoder_generated_adapt"
                if DECODER_GENERATED_ADAPT_ONLY
                else
                "riemannian_metric_flow_decoder_body_adapt"
                if DECODER_ADAPT and DECODER_ADAPT_BERT_LAST_N_LAYERS > 0
                else "riemannian_metric_flow_decoder_adapt"
                if DECODER_ADAPT
                else "riemannian_metric_flow"
            ),
        }, checkpoint_path)
        print(
            f"saved best model at val score {best_score:.4f} | "
            f"flow loss {avg_val_loss:.4f} | path {checkpoint_path}\n",
            flush=True,
        )
