import os
import random
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DownloadConfig, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer

sys.path.insert(0, ".")
from parallel_decoder import BertEncoder, ParallelDecoder, cached_from_pretrained

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# -- config --------------------------------------------------------------------
RESUME = False
SEED = 1337
PROMPT_LEN = 16
COND_DROP_PROB = 0.10
MAX_SEQ_LEN = 32
BASE_NOISE_STD = 0.30
CALIBRATE_GENERATED_LATENTS = True
TARGET_LATENT_MEAN = -0.003
TARGET_LATENT_STD = 0.264
COMPILE_MODELS = False
FAST_DEBUG = False
TRAIN_SIZE = 100000
TRAIN_BATCH_SIZE = 512
DATALOADER_NUM_WORKERS = 4
FLOW_HIDDEN_DIM = 512
FLOW_DEPTH = 5
METRIC_HIDDEN_DIM = 256
LOG_EVERY = 50
METRIC_LOSS_WEIGHT = 1.0
EUCLIDEAN_LOSS_WEIGHT = 0.05
X0_LOSS_WEIGHT = 0.25
DECODE_LOSS_WEIGHT = 0.05
DECODE_LOSS_BATCH = 128
ROLLOUT_LOSS_WEIGHT = 0.25
ROLLOUT_DECODE_LOSS_WEIGHT = 0.05
ROLLOUT_HIDDEN_LOSS_WEIGHT = 0.0
ROLLOUT_LOGIT_KL_WEIGHT = 0.0
ROLLOUT_LOGIT_KL_TEMP = 2.0
ROLLOUT_ENTROPY_LOSS_WEIGHT = 0.005
ROLLOUT_ENTROPY_MARGIN = 0.0
ROLLOUT_ENTROPY_FULL_EPOCHS = 20
ROLLOUT_ENTROPY_DECAY_EPOCHS = 0
ROLLOUT_GATED_GEN_CE_WEIGHT = 0.01
ROLLOUT_GATED_GEN_CE_TOP1_CAP = 0.25
ROLLOUT_GATED_GEN_CE_ENTROPY_MARGIN = 1.0
ROLLOUT_TARGET_PROB_WEIGHT = 0.005
ROLLOUT_TARGET_PROB_MARGIN = 0.20
ROLLOUT_TARGET_PROB_TOP1_CAP = 0.50
ROLLOUT_LOGIT_BALANCE_WEIGHT = 0.0
ROLLOUT_LOGIT_BALANCE_TOPK = 8
ROLLOUT_LOGIT_BALANCE_TARGET = 0.08
LOGIT_BALANCE_SPECIAL_IDS = (0, 100, 101, 102, 103)
ROLLOUT_NORM_LOSS_WEIGHT = 0.025
ROLLOUT_DIVERSITY_LOSS_WEIGHT = 0.01
ROLLOUT_DIVERSITY_MAX_TOKENS = 512
ROLLOUT_BATCH = 64
ROLLOUT_TRAIN_STEPS = 8
RAW_NORM_GAP_SCORE_WEIGHT = 0.05
COLLAPSE_UNIQ_TARGET = 0.25
COLLAPSE_MAXFRAC_TARGET = 0.50
COLLAPSE_UNIQ_SCORE_WEIGHT = 2.0
COLLAPSE_MAXFRAC_SCORE_WEIGHT = 2.0
METRIC_REG = 1e-4
METRIC_WARMUP_REG_MULT = 100.0
METRIC_WARMUP_STEPS = 1000
METRIC_LOG_BOUND = 0.50
SELF_GATE_SCALE = 0.10
CROSS_GATE_SCALE = 0.10
GATE_INIT = 0.50
GATE_REG_WEIGHT = 0.0
GATE_LR_MULT = 20.0
ODE_STEPS = 16
EVAL_SAMPLE_TEMPERATURE = 0.8
EVAL_SAMPLE_TOP_K = 50
EVAL_SAMPLE_TOP_P = 0.95
DECODER_ADAPT = True
DECODER_ADAPT_LR = 1e-6
DECODER_ADAPT_REAL_CE_WEIGHT = 0.10
DECODER_ADAPT_GEN_CE_WEIGHT = 0.05
DECODER_ADAPT_GEN_CE_RAMP_EPOCHS = 2
DECODER_ADAPT_PRESERVE_KL_WEIGHT = 0.10
DECODER_ADAPT_KL_TEMP = 2.0
DECODER_ADAPT_MODULES = ("project_up", "to_logits")
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using: {device}")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


seed_everything(SEED)
print(f"seed: {SEED}", flush=True)


def atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


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


def build_stage2_dataloaders(tokenizer, train_size, batch_size, max_length):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    try:
        ds = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            download_config=DownloadConfig(local_files_only=True),
        )
        print("loaded wikitext from local datasets cache", flush=True)
    except Exception as exc:
        print(f"local wikitext cache unavailable ({exc}) | trying online load", flush=True)
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_size = min(train_size, len(ds["train"]))
    small_train = ds["train"].select(range(train_size))
    small_val = ds["validation"]

    small_train = small_train.filter(lambda x: len(x["text"].strip()) > 10)
    small_val = small_val.filter(lambda x: len(x["text"].strip()) > 10)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    train_tok = small_train.map(tokenize, batched=True)
    val_tok = small_val.map(tokenize, batched=True)
    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(
        train_tok,
        batch_size=batch_size,
        shuffle=True,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=DATALOADER_NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_tok,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=DATALOADER_NUM_WORKERS > 0,
    )
    print(
        f"train batches: {len(train_loader)}  val batches: {len(val_loader)}  "
        f"max_length: {max_length}",
        flush=True,
    )
    return train_loader, val_loader


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
        self.pos_proj = nn.Linear(1, hidden_dim)
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
                "self_norm": nn.LayerNorm(hidden_dim),
                "self_attn": nn.MultiheadAttention(
                    hidden_dim,
                    num_heads=8,
                    batch_first=True,
                ),
                "cross_norm": nn.LayerNorm(hidden_dim),
                "cross_attn": nn.MultiheadAttention(
                    hidden_dim,
                    num_heads=8,
                    batch_first=True,
                ),
            })
            for _ in range(depth)
        ])
        self.self_gates = nn.ParameterList([nn.Parameter(torch.tensor(GATE_INIT)) for _ in range(depth)])
        self.cross_gates = nn.ParameterList([nn.Parameter(torch.tensor(GATE_INIT)) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

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

        prompt_h = None
        prompt_key_padding_mask = None
        if z_cond.dim() == 3 and z_cond.size(1) == PROMPT_LEN:
            prompt_h = self.prompt_proj(z_cond) + self.prompt_pos
            cond = self.cond_proj(prompt_h.reshape(prompt_h.size(0), -1))
            prompt_key_padding_mask = z_cond.abs().sum(dim=-1) == 0
            if prompt_key_padding_mask.all(dim=1).any():
                prompt_key_padding_mask = prompt_key_padding_mask.clone()
                prompt_key_padding_mask[prompt_key_padding_mask.all(dim=1), 0] = False
        elif z_cond.dim() == 3:
            cond = z_cond.mean(dim=1)
        else:
            cond = z_cond
        cond = cond.unsqueeze(1).expand(-1, z_t.size(1), -1)

        inp = torch.cat([z_t, cond, t.unsqueeze(-1), pos.unsqueeze(-1)], dim=-1)
        h = self.in_proj(inp) + self.pos_proj(pos.unsqueeze(-1))
        if mask is not None:
            h = h * mask.to(h.dtype).unsqueeze(-1)
            self_key_padding_mask = mask == 0
            if self_key_padding_mask.all(dim=1).any():
                self_key_padding_mask = self_key_padding_mask.clone()
                self_key_padding_mask[self_key_padding_mask.all(dim=1), 0] = False
        else:
            self_key_padding_mask = None

        for block_idx, block in enumerate(self.blocks):
            residual = h
            x = block["norm"](h)
            x = block["conv"](x.transpose(1, 2)).transpose(1, 2)
            x = block["mix"](x)
            h = residual + x
            self_in = block["self_norm"](h)
            self_out, _ = block["self_attn"](
                self_in,
                self_in,
                self_in,
                key_padding_mask=self_key_padding_mask,
                need_weights=False,
            )
            h = h + SELF_GATE_SCALE * self.self_gates[block_idx].tanh() * self_out
            if prompt_h is not None:
                cross_in = block["cross_norm"](h)
                cross_out, _ = block["cross_attn"](
                    cross_in,
                    prompt_h,
                    prompt_h,
                    key_padding_mask=prompt_key_padding_mask,
                    need_weights=False,
                )
                h = h + CROSS_GATE_SCALE * self.cross_gates[block_idx].tanh() * cross_out
            if mask is not None:
                h = h * mask.to(h.dtype).unsqueeze(-1)

        out = self.out_proj(self.out_norm(h))
        if squeeze:
            out = out.squeeze(1)
        return out


class MetricNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=METRIC_HIDDEN_DIM, log_bound=METRIC_LOG_BOUND):
        super().__init__()
        self.log_bound = log_bound
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t, t, z_cond, pos):
        inp = torch.cat([z_t, z_cond, t.unsqueeze(-1), pos.unsqueeze(-1)], dim=-1)
        log_g = self.net(inp)
        log_g = log_g - log_g.mean(dim=-1, keepdim=True)
        log_g = self.log_bound * torch.tanh(log_g / self.log_bound)
        g_diag = torch.exp(log_g)
        return g_diag / g_diag.mean(dim=-1, keepdim=True).clamp_min(1e-6)


def prompt_condition(z_data, attention_mask, prompt_len=PROMPT_LEN):
    prompt_z = z_data[:, :prompt_len, :]
    prompt_mask = attention_mask[:, :prompt_len].to(prompt_z.dtype).unsqueeze(-1)
    return prompt_z * prompt_mask


def suffix_positions(batch_size, suffix_len, device, dtype=torch.float32):
    pos = torch.arange(PROMPT_LEN, PROMPT_LEN + suffix_len, device=device, dtype=dtype)
    pos = pos / max(MAX_SEQ_LEN - 1, 1)
    return pos.unsqueeze(0).expand(batch_size, suffix_len)


def attention_gate_stats(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)
    return (
        SELF_GATE_SCALE * torch.stack([gate.detach().tanh().abs() for gate in model.self_gates]).mean().item(),
        CROSS_GATE_SCALE * torch.stack([gate.detach().tanh().abs() for gate in model.cross_gates]).mean().item(),
    )


def attention_gate_regularizer(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)
    self_reg = torch.stack([gate.tanh().pow(2) for gate in model.self_gates]).mean()
    cross_reg = torch.stack([gate.tanh().pow(2) for gate in model.cross_gates]).mean()
    return GATE_REG_WEIGHT * (self_reg + cross_reg)


def attention_gate_parameters(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)
    return list(model.self_gates.parameters()) + list(model.cross_gates.parameters())


def non_gate_flow_parameters(flow_net):
    gate_param_ids = {id(param) for param in attention_gate_parameters(flow_net)}
    return [param for param in flow_net.parameters() if id(param) not in gate_param_ids]


def attention_gate_grad_stats(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)

    def mean_abs_grad(gates):
        grads = [gate.grad.detach().abs().mean() for gate in gates if gate.grad is not None]
        if not grads:
            return 0.0
        return torch.stack(grads).mean().item()

    return mean_abs_grad(model.self_gates), mean_abs_grad(model.cross_gates)


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


def calibrate_latents(z, mask=None, target_mean=TARGET_LATENT_MEAN, target_std=TARGET_LATENT_STD, eps=1e-6):
    if not CALIBRATE_GENERATED_LATENTS:
        return z
    if mask is not None:
        valid = mask.bool()
        if valid.any():
            mean = z[valid].mean()
            std = z[valid].std().clamp_min(eps)
        else:
            mean = z.mean()
            std = z.std().clamp_min(eps)
    else:
        mean = z.mean()
        std = z.std().clamp_min(eps)
    return (z - mean) * (target_std / std) + target_mean


def flatten_valid(z_target, z_t, v_true, v_pred, z_x0, z_cond, pos, t, target_mask):
    B, T, D = z_target.shape
    pooled_cond = z_cond.mean(dim=1)
    cond_flat = pooled_cond.unsqueeze(1).expand(-1, T, -1).reshape(B * T, D)
    z_target = z_target.reshape(B * T, D)
    z_t = z_t.reshape(B * T, D)
    v_true = v_true.reshape(B * T, D)
    v_pred = v_pred.reshape(B * T, D)
    z_x0 = z_x0.reshape(B * T, D)
    pos = pos.reshape(B * T)
    t = t.reshape(B * T)
    if target_mask is not None:
        valid = target_mask.reshape(B * T).bool()
        z_target = z_target[valid]
        z_t = z_t[valid]
        v_true = v_true[valid]
        v_pred = v_pred[valid]
        z_x0 = z_x0[valid]
        cond_flat = cond_flat[valid]
        pos = pos[valid]
        t = t[valid]
    return z_target, z_t, v_true, v_pred, z_x0, cond_flat, pos, t


def decoder_hidden_from_latent(decoder, z_latent):
    x = decoder.project_up(z_latent)
    out = decoder.bert(inputs_embeds=x)
    return out.last_hidden_state


def valid_token_latents(z, mask=None):
    if mask is None:
        return z.reshape(-1, z.size(-1))
    return z[mask.bool()]


def pairwise_distance_match_loss(z_pred, z_target, mask=None, max_tokens=ROLLOUT_DIVERSITY_MAX_TOKENS, eps=1e-6):
    pred_tokens = valid_token_latents(z_pred, mask)
    target_tokens = valid_token_latents(z_target, mask)
    n_tokens = min(pred_tokens.size(0), target_tokens.size(0))
    if n_tokens < 2:
        return z_pred.new_tensor(0.0)
    pred_tokens = pred_tokens[:n_tokens]
    target_tokens = target_tokens[:n_tokens]
    if n_tokens > max_tokens:
        sample_idx = torch.randperm(n_tokens, device=z_pred.device)[:max_tokens]
        pred_tokens = pred_tokens[sample_idx]
        target_tokens = target_tokens[sample_idx]
    pred_dist = torch.pdist(pred_tokens.float(), p=2)
    target_dist = torch.pdist(target_tokens.detach().float(), p=2)
    scale = target_dist.mean().clamp_min(eps)
    return F.smooth_l1_loss(pred_dist / scale, target_dist / scale)


def logit_balance_loss(
    logits,
    mask=None,
    topk=ROLLOUT_LOGIT_BALANCE_TOPK,
    target=ROLLOUT_LOGIT_BALANCE_TARGET,
):
    suffix_logits = logits[:, PROMPT_LEN:, :].float()
    probs = suffix_logits.softmax(dim=-1)
    if mask is not None:
        valid = mask.bool()
        if not valid.any():
            return logits.new_tensor(0.0), 0.0
        probs = probs[valid]
    else:
        probs = probs.reshape(-1, probs.size(-1))
    probs = probs.clone()
    special_ids = torch.tensor(LOGIT_BALANCE_SPECIAL_IDS, device=probs.device)
    probs[:, special_ids] = 0.0
    mean_probs = probs.mean(dim=0)
    top_mass = mean_probs.topk(min(topk, mean_probs.numel())).values.sum()
    return (top_mass - target).clamp_min(0.0).pow(2), top_mass.detach().item()


def entropy_weight_multiplier(global_step=None, steps_per_epoch=None):
    if global_step is None or not steps_per_epoch:
        return 1.0
    full_steps = max(0, int(ROLLOUT_ENTROPY_FULL_EPOCHS * steps_per_epoch))
    decay_steps = max(0, int(ROLLOUT_ENTROPY_DECAY_EPOCHS * steps_per_epoch))
    if global_step < full_steps:
        return 1.0
    if decay_steps <= 0:
        return 0.0
    return max(0.0, 1.0 - (global_step - full_steps) / decay_steps)


def entropy_gap_loss(gen_logits, oracle_logits, mask=None, margin=ROLLOUT_ENTROPY_MARGIN):
    gen_probs = gen_logits[:, PROMPT_LEN:, :].float().softmax(dim=-1)
    oracle_probs = oracle_logits[:, PROMPT_LEN:, :].float().softmax(dim=-1)
    gen_entropy = -(gen_probs * gen_probs.clamp_min(1e-9).log()).sum(dim=-1)
    oracle_entropy = -(oracle_probs * oracle_probs.clamp_min(1e-9).log()).sum(dim=-1)
    if mask is not None:
        valid = mask.bool()
        if not valid.any():
            return gen_logits.new_tensor(0.0), 0.0, 0.0
        gen_entropy = gen_entropy[valid]
        oracle_entropy = oracle_entropy[valid]
    excess_entropy = (gen_entropy - oracle_entropy.detach() - margin).clamp_min(0.0)
    return (
        F.smooth_l1_loss(excess_entropy, torch.zeros_like(excess_entropy)),
        gen_entropy.detach().mean().item(),
        oracle_entropy.detach().mean().item(),
    )


def gated_generated_ce_loss(
    gen_logits,
    oracle_logits,
    suffix_ids,
    mask=None,
    entropy_margin=ROLLOUT_GATED_GEN_CE_ENTROPY_MARGIN,
    top1_cap=ROLLOUT_GATED_GEN_CE_TOP1_CAP,
):
    suffix_logits = gen_logits[:, PROMPT_LEN:, :].float()
    oracle_probs = oracle_logits[:, PROMPT_LEN:, :].float().softmax(dim=-1)
    gen_probs = suffix_logits.softmax(dim=-1)
    gen_entropy = -(gen_probs * gen_probs.clamp_min(1e-9).log()).sum(dim=-1)
    oracle_entropy = -(oracle_probs * oracle_probs.clamp_min(1e-9).log()).sum(dim=-1)
    gen_top1 = gen_probs.max(dim=-1).values

    suffix_targets = suffix_ids[:gen_logits.size(0)]
    valid = suffix_targets != 0
    if mask is not None:
        valid = valid & mask.bool()
    if not valid.any():
        return gen_logits.new_tensor(0.0), 0.0, 0.0

    active = (
        (gen_entropy > oracle_entropy.detach() + entropy_margin)
        & (gen_top1 < top1_cap)
        & valid
    )
    mean_top1 = gen_top1[valid].detach().mean().item()
    active_frac = active.float()[valid].mean().item()
    if not active.any():
        return gen_logits.new_tensor(0.0), active_frac, mean_top1

    token_ce = F.cross_entropy(
        suffix_logits.reshape(-1, suffix_logits.size(-1)),
        suffix_targets.reshape(-1),
        ignore_index=0,
        reduction="none",
    ).reshape_as(gen_entropy)
    return token_ce[active].mean(), active_frac, mean_top1


def target_probability_loss(
    gen_logits,
    oracle_logits,
    suffix_ids,
    mask=None,
    margin=ROLLOUT_TARGET_PROB_MARGIN,
    top1_cap=ROLLOUT_TARGET_PROB_TOP1_CAP,
):
    suffix_logits = gen_logits[:, PROMPT_LEN:, :].float()
    gen_probs = suffix_logits.softmax(dim=-1)
    with torch.no_grad():
        oracle_probs = oracle_logits[:, PROMPT_LEN:, :].float().softmax(dim=-1)
    gen_top1 = gen_probs.max(dim=-1).values

    suffix_targets = suffix_ids[:gen_logits.size(0)]
    valid = suffix_targets != 0
    if mask is not None:
        valid = valid & mask.bool()
    if not valid.any():
        return gen_logits.new_tensor(0.0), 0.0, 0.0, 0.0

    target_gather_ids = suffix_targets.clamp(0, gen_probs.size(-1) - 1).unsqueeze(-1)
    gen_target_prob = gen_probs.gather(dim=-1, index=target_gather_ids).squeeze(-1)
    oracle_target_prob = oracle_probs.gather(dim=-1, index=target_gather_ids).squeeze(-1)
    active = (
        ((oracle_target_prob.detach() - gen_target_prob) > margin)
        & (gen_top1 < top1_cap)
        & valid
    )
    active_frac = active.float()[valid].mean().item()
    mean_gen_target_prob = gen_target_prob[valid].detach().mean().item()
    mean_oracle_target_prob = oracle_target_prob[valid].detach().mean().item()
    if not active.any():
        return (
            gen_logits.new_tensor(0.0),
            active_frac,
            mean_gen_target_prob,
            mean_oracle_target_prob,
        )

    token_ce = F.cross_entropy(
        suffix_logits.reshape(-1, suffix_logits.size(-1)),
        suffix_targets.reshape(-1),
        ignore_index=0,
        reduction="none",
    ).reshape_as(gen_target_prob)
    return (
        token_ce[active].mean(),
        active_frac,
        mean_gen_target_prob,
        mean_oracle_target_prob,
    )


def sample_token_ids(
    logits,
    tokenizer,
    temperature=EVAL_SAMPLE_TEMPERATURE,
    top_k=EVAL_SAMPLE_TOP_K,
    top_p=EVAL_SAMPLE_TOP_P,
):
    if temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits.float() / temperature
    for token_id in tokenizer.all_special_ids:
        logits[..., token_id] = -float("inf")

    if top_k is not None and top_k > 0:
        kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, -float("inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        keep = sorted_probs.cumsum(dim=-1) <= top_p
        keep[..., 0] = True
        sorted_logits = sorted_logits.masked_fill(~keep, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(dim=-1, index=sorted_idx, src=sorted_logits)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).view(logits.shape[:-1])


def flow_matching_loss(
    flow_net,
    metric_net,
    z_target,
    z_cond,
    target_mask=None,
    decoder=None,
    z_prompt=None,
    suffix_ids=None,
    teacher_decoder=None,
    global_step=None,
    steps_per_epoch=None,
    return_stats=False,
):
    if target_mask is not None:
        has_target = target_mask.sum(dim=1) > 0
        if not has_target.any():
            zero = next(flow_net.parameters()).sum() + next(metric_net.parameters()).sum()
            if return_stats:
                return zero * 0.0, {
                    "metric_loss": 0.0,
                    "euclidean_loss": 0.0,
                    "x0_loss": 0.0,
                    "decode_loss": 0.0,
                    "weighted_decode_loss": 0.0,
                    "metric_mean": 0.0,
                    "metric_std": 0.0,
                    "metric_min": 0.0,
                    "metric_max": 0.0,
                    "metric_reg": 0.0,
                    "metric_reg_mult": 1.0,
                    "rollout_loss": 0.0,
                    "rollout_decode_loss": 0.0,
                    "weighted_rollout_decode_loss": 0.0,
                    "rollout_hidden_loss": 0.0,
                    "weighted_rollout_hidden_loss": 0.0,
                    "rollout_logit_kl": 0.0,
                    "weighted_rollout_logit_kl": 0.0,
                    "rollout_entropy_loss": 0.0,
                    "weighted_rollout_entropy_loss": 0.0,
                    "rollout_entropy_mult": 0.0,
                    "rollout_gen_entropy": 0.0,
                    "rollout_oracle_entropy": 0.0,
                    "rollout_gated_gen_ce": 0.0,
                    "weighted_rollout_gated_gen_ce": 0.0,
                    "rollout_gated_gen_ce_active": 0.0,
                    "rollout_gated_gen_ce_top1": 0.0,
                    "rollout_target_prob_loss": 0.0,
                    "weighted_rollout_target_prob_loss": 0.0,
                    "rollout_target_prob_active": 0.0,
                    "rollout_target_prob_gen": 0.0,
                    "rollout_target_prob_oracle": 0.0,
                    "rollout_logit_balance": 0.0,
                    "weighted_rollout_logit_balance": 0.0,
                    "rollout_logit_topmass": 0.0,
                    "rollout_norm_loss": 0.0,
                    "rollout_diversity_loss": 0.0,
                    "weighted_rollout_diversity_loss": 0.0,
                    "decoder_adapt_real_ce": 0.0,
                    "weighted_decoder_adapt_real_ce": 0.0,
                    "decoder_adapt_gen_ce": 0.0,
                    "decoder_adapt_gen_ce_mult": 0.0,
                    "weighted_decoder_adapt_gen_ce": 0.0,
                    "decoder_adapt_preserve_kl": 0.0,
                    "weighted_decoder_adapt_preserve_kl": 0.0,
                    "self_gate": 0.0,
                    "cross_gate": 0.0,
                    "gate_reg": 0.0,
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
    pos_seq = suffix_positions(B, T, z_target.device, z_target.dtype)
    z_noise = torch.randn_like(z_target) * BASE_NOISE_STD
    t_seq = torch.rand(B, T, device=z_target.device).pow(2)
    z_t = (1 - t_seq.unsqueeze(-1)) * z_noise + t_seq.unsqueeze(-1) * z_target
    v_true = z_target - z_noise
    v_pred = flow_net(z_t, t_seq, z_cond, pos_seq, target_mask)
    z_x0 = z_t + (1.0 - t_seq.unsqueeze(-1)) * v_pred

    z_flat, z_t_flat, v_true_flat, v_pred_flat, z_x0_flat, cond_flat, pos_flat, t_flat = flatten_valid(
        z_target,
        z_t,
        v_true,
        v_pred,
        z_x0,
        z_cond,
        pos_seq,
        t_seq,
        target_mask,
    )

    g_diag = metric_net(z_t_flat, t_flat, cond_flat, pos_flat)
    err = (v_pred_flat - v_true_flat).pow(2)
    metric_loss = (g_diag * err).mean(dim=-1).mean()
    euclidean_loss = err.mean()
    x0_loss = F.mse_loss(z_x0_flat, z_flat)
    if global_step is None or METRIC_WARMUP_STEPS <= 0:
        metric_reg_mult = 1.0
    else:
        warmup_left = max(0.0, 1.0 - global_step / METRIC_WARMUP_STEPS)
        metric_reg_mult = 1.0 + (METRIC_WARMUP_REG_MULT - 1.0) * warmup_left
    metric_reg = (METRIC_REG * metric_reg_mult) * g_diag.log().pow(2).mean()
    gate_reg = attention_gate_regularizer(flow_net)

    decode_loss = z_target.new_tensor(0.0)
    if decoder is not None and z_prompt is not None and suffix_ids is not None and DECODE_LOSS_WEIGHT > 0:
        n_decode = min(DECODE_LOSS_BATCH, B)
        z_pred_seq = torch.cat([z_prompt[:n_decode], z_x0[:n_decode]], dim=1)
        logits = decoder.decode_from_latent(z_pred_seq)
        suffix_logits = logits[:, PROMPT_LEN:, :].reshape(-1, logits.size(-1))
        suffix_targets = suffix_ids[:n_decode].reshape(-1)
        decode_loss = F.cross_entropy(suffix_logits, suffix_targets, ignore_index=0)

    rollout_loss = z_target.new_tensor(0.0)
    rollout_decode_loss = z_target.new_tensor(0.0)
    rollout_hidden_loss = z_target.new_tensor(0.0)
    rollout_logit_kl = z_target.new_tensor(0.0)
    rollout_entropy_loss = z_target.new_tensor(0.0)
    rollout_entropy_mult = entropy_weight_multiplier(global_step, steps_per_epoch)
    rollout_gen_entropy = 0.0
    rollout_oracle_entropy = 0.0
    rollout_gated_gen_ce = z_target.new_tensor(0.0)
    rollout_gated_gen_ce_active = 0.0
    rollout_gated_gen_ce_top1 = 0.0
    rollout_target_prob_loss = z_target.new_tensor(0.0)
    rollout_target_prob_active = 0.0
    rollout_target_prob_gen = 0.0
    rollout_target_prob_oracle = 0.0
    rollout_logit_balance = z_target.new_tensor(0.0)
    rollout_logit_topmass = 0.0
    rollout_norm_loss = z_target.new_tensor(0.0)
    rollout_diversity_loss = z_target.new_tensor(0.0)
    decoder_adapt_real_ce = z_target.new_tensor(0.0)
    decoder_adapt_gen_ce = z_target.new_tensor(0.0)
    decoder_adapt_preserve_kl = z_target.new_tensor(0.0)
    decoder_adapt_gen_ce_mult = 1.0
    if DECODER_ADAPT_GEN_CE_RAMP_EPOCHS > 0 and global_step is not None and steps_per_epoch:
        ramp_steps = max(1, int(DECODER_ADAPT_GEN_CE_RAMP_EPOCHS * steps_per_epoch))
        decoder_adapt_gen_ce_mult = min(1.0, (global_step + 1) / ramp_steps)
    if ROLLOUT_LOSS_WEIGHT > 0 and ROLLOUT_TRAIN_STEPS > 0:
        n_rollout = min(ROLLOUT_BATCH, B)
        z_roll_target = z_target[:n_rollout]
        z_roll_cond = z_cond[:n_rollout]
        roll_mask = target_mask[:n_rollout] if target_mask is not None else None
        pos_roll = suffix_positions(n_rollout, T, z_target.device, z_target.dtype)
        z_roll = torch.randn_like(z_roll_target) * BASE_NOISE_STD
        dt = 1.0 / ROLLOUT_TRAIN_STEPS
        for i in range(ROLLOUT_TRAIN_STEPS):
            t_roll = torch.full((n_rollout, T), i / ROLLOUT_TRAIN_STEPS, device=z_target.device)
            v_roll, _ = natural_velocity(flow_net, metric_net, z_roll, t_roll, z_roll_cond, pos_roll)
            z_roll = z_roll + v_roll * dt
            if roll_mask is not None:
                z_roll = z_roll * roll_mask.to(z_roll.dtype).unsqueeze(-1)

        if roll_mask is not None:
            valid_roll = roll_mask.bool()
            if valid_roll.any():
                rollout_loss = F.mse_loss(z_roll[valid_roll], z_roll_target[valid_roll])
                rollout_norm_loss = F.mse_loss(
                    z_roll[valid_roll].norm(dim=-1),
                    z_roll_target[valid_roll].norm(dim=-1),
                )
        else:
            rollout_loss = F.mse_loss(z_roll, z_roll_target)
            rollout_norm_loss = F.mse_loss(z_roll.norm(dim=-1), z_roll_target.norm(dim=-1))

        if ROLLOUT_DIVERSITY_LOSS_WEIGHT > 0:
            rollout_diversity_loss = pairwise_distance_match_loss(z_roll, z_roll_target, roll_mask)

        need_entropy_logits = (
            decoder is not None
            and z_prompt is not None
            and (
                (ROLLOUT_ENTROPY_LOSS_WEIGHT > 0 and rollout_entropy_mult > 0)
                or (ROLLOUT_GATED_GEN_CE_WEIGHT > 0 and suffix_ids is not None)
                or (ROLLOUT_TARGET_PROB_WEIGHT > 0 and suffix_ids is not None)
            )
        )
        if need_entropy_logits:
            entropy_decoder = teacher_decoder if teacher_decoder is not None else decoder
            z_entropy_gen_seq = torch.cat([z_prompt[:n_rollout], z_roll], dim=1)
            z_entropy_real_seq = torch.cat([z_prompt[:n_rollout], z_roll_target], dim=1)
            gen_entropy_logits = entropy_decoder.decode_from_latent(z_entropy_gen_seq)
            with torch.no_grad():
                oracle_entropy_logits = entropy_decoder.decode_from_latent(z_entropy_real_seq)
            if ROLLOUT_ENTROPY_LOSS_WEIGHT > 0 and rollout_entropy_mult > 0:
                rollout_entropy_loss, rollout_gen_entropy, rollout_oracle_entropy = entropy_gap_loss(
                    gen_entropy_logits,
                    oracle_entropy_logits,
                    roll_mask,
                )
            if ROLLOUT_GATED_GEN_CE_WEIGHT > 0 and suffix_ids is not None:
                rollout_gated_gen_ce, rollout_gated_gen_ce_active, rollout_gated_gen_ce_top1 = gated_generated_ce_loss(
                    gen_entropy_logits,
                    oracle_entropy_logits,
                    suffix_ids[:n_rollout],
                    roll_mask,
                )
            if ROLLOUT_TARGET_PROB_WEIGHT > 0 and suffix_ids is not None:
                (
                    rollout_target_prob_loss,
                    rollout_target_prob_active,
                    rollout_target_prob_gen,
                    rollout_target_prob_oracle,
                ) = target_probability_loss(
                    gen_entropy_logits,
                    oracle_entropy_logits,
                    suffix_ids[:n_rollout],
                    roll_mask,
                )

        if (
            decoder is not None
            and z_prompt is not None
            and suffix_ids is not None
            and (
                ROLLOUT_DECODE_LOSS_WEIGHT > 0
                or ROLLOUT_HIDDEN_LOSS_WEIGHT > 0
                or ROLLOUT_LOGIT_KL_WEIGHT > 0
            )
        ):
            z_roll_seq = torch.cat([z_prompt[:n_rollout], z_roll], dim=1)
            roll_hidden = decoder_hidden_from_latent(decoder, z_roll_seq)
            roll_logits = None

            if ROLLOUT_DECODE_LOSS_WEIGHT > 0:
                roll_logits = decoder.to_logits(roll_hidden)
                roll_suffix_logits = roll_logits[:, PROMPT_LEN:, :].reshape(-1, roll_logits.size(-1))
                roll_suffix_targets = suffix_ids[:n_rollout].reshape(-1)
                rollout_decode_loss = F.cross_entropy(
                    roll_suffix_logits,
                    roll_suffix_targets,
                    ignore_index=0,
                )

            if ROLLOUT_HIDDEN_LOSS_WEIGHT > 0 or ROLLOUT_LOGIT_KL_WEIGHT > 0:
                z_real_seq = torch.cat([z_prompt[:n_rollout], z_roll_target], dim=1)
                with torch.no_grad():
                    real_hidden = decoder_hidden_from_latent(decoder, z_real_seq)

            if ROLLOUT_HIDDEN_LOSS_WEIGHT > 0:
                if roll_mask is not None:
                    valid_hidden = roll_mask.bool()
                    if valid_hidden.any():
                        rollout_hidden_loss = F.mse_loss(
                            roll_hidden[:, PROMPT_LEN:, :][valid_hidden],
                            real_hidden[:, PROMPT_LEN:, :][valid_hidden],
                        )
                else:
                    rollout_hidden_loss = F.mse_loss(
                        roll_hidden[:, PROMPT_LEN:, :],
                        real_hidden[:, PROMPT_LEN:, :],
                    )

            if ROLLOUT_LOGIT_KL_WEIGHT > 0:
                if roll_logits is None:
                    roll_logits = decoder.to_logits(roll_hidden)
                with torch.no_grad():
                    real_logits = decoder.to_logits(real_hidden)
                roll_suffix_logits = roll_logits[:, PROMPT_LEN:, :].float()
                real_suffix_logits = real_logits[:, PROMPT_LEN:, :].float()
                temp = ROLLOUT_LOGIT_KL_TEMP
                token_kl = F.kl_div(
                    F.log_softmax(roll_suffix_logits / temp, dim=-1),
                    F.softmax(real_suffix_logits / temp, dim=-1),
                    reduction="none",
                ).sum(dim=-1) * (temp * temp)
                if roll_mask is not None:
                    valid_kl = roll_mask.bool()
                    if valid_kl.any():
                        rollout_logit_kl = token_kl[valid_kl].mean()
                else:
                    rollout_logit_kl = token_kl.mean()

        if DECODER_ADAPT and decoder is not None and z_prompt is not None and suffix_ids is not None:
            z_real_seq = torch.cat([z_prompt[:n_rollout], z_roll_target], dim=1)
            z_gen_seq = torch.cat([z_prompt[:n_rollout], z_roll], dim=1)
            suffix_targets = suffix_ids[:n_rollout].reshape(-1)

            if DECODER_ADAPT_REAL_CE_WEIGHT > 0:
                real_logits = decoder.decode_from_latent(z_real_seq)
                decoder_adapt_real_ce = F.cross_entropy(
                    real_logits[:, PROMPT_LEN:, :].reshape(-1, real_logits.size(-1)),
                    suffix_targets,
                    ignore_index=0,
                )
            else:
                real_logits = None

            if DECODER_ADAPT_GEN_CE_WEIGHT > 0:
                gen_logits = decoder.decode_from_latent(z_gen_seq)
                decoder_adapt_gen_ce = F.cross_entropy(
                    gen_logits[:, PROMPT_LEN:, :].reshape(-1, gen_logits.size(-1)),
                    suffix_targets,
                    ignore_index=0,
                )
            else:
                gen_logits = None

            if ROLLOUT_LOGIT_BALANCE_WEIGHT > 0:
                if gen_logits is None:
                    gen_logits = decoder.decode_from_latent(z_gen_seq)
                rollout_logit_balance, rollout_logit_topmass = logit_balance_loss(gen_logits, roll_mask)

            if DECODER_ADAPT_PRESERVE_KL_WEIGHT > 0 and teacher_decoder is not None:
                if real_logits is None:
                    real_logits = decoder.decode_from_latent(z_real_seq)
                with torch.no_grad():
                    teacher_logits = teacher_decoder.decode_from_latent(z_real_seq)
                temp = DECODER_ADAPT_KL_TEMP
                token_kl = F.kl_div(
                    F.log_softmax(real_logits[:, PROMPT_LEN:, :].float() / temp, dim=-1),
                    F.softmax(teacher_logits[:, PROMPT_LEN:, :].float() / temp, dim=-1),
                    reduction="none",
                ).sum(dim=-1) * (temp * temp)
                if roll_mask is not None:
                    valid_kl = roll_mask.bool()
                    if valid_kl.any():
                        decoder_adapt_preserve_kl = token_kl[valid_kl].mean()
                else:
                    decoder_adapt_preserve_kl = token_kl.mean()

    total_loss = (
        METRIC_LOSS_WEIGHT * metric_loss
        + EUCLIDEAN_LOSS_WEIGHT * euclidean_loss
        + X0_LOSS_WEIGHT * x0_loss
        + DECODE_LOSS_WEIGHT * decode_loss
        + ROLLOUT_LOSS_WEIGHT * rollout_loss
        + ROLLOUT_DECODE_LOSS_WEIGHT * rollout_decode_loss
        + ROLLOUT_HIDDEN_LOSS_WEIGHT * rollout_hidden_loss
        + ROLLOUT_LOGIT_KL_WEIGHT * rollout_logit_kl
        + (ROLLOUT_ENTROPY_LOSS_WEIGHT * rollout_entropy_mult) * rollout_entropy_loss
        + ROLLOUT_GATED_GEN_CE_WEIGHT * rollout_gated_gen_ce
        + ROLLOUT_TARGET_PROB_WEIGHT * rollout_target_prob_loss
        + ROLLOUT_LOGIT_BALANCE_WEIGHT * rollout_logit_balance
        + ROLLOUT_NORM_LOSS_WEIGHT * rollout_norm_loss
        + ROLLOUT_DIVERSITY_LOSS_WEIGHT * rollout_diversity_loss
        + DECODER_ADAPT_REAL_CE_WEIGHT * decoder_adapt_real_ce
        + (DECODER_ADAPT_GEN_CE_WEIGHT * decoder_adapt_gen_ce_mult) * decoder_adapt_gen_ce
        + DECODER_ADAPT_PRESERVE_KL_WEIGHT * decoder_adapt_preserve_kl
        + metric_reg
        + gate_reg
    )
    if return_stats:
        self_gate, cross_gate = attention_gate_stats(flow_net)
        return total_loss, {
            "metric_loss": metric_loss.detach().item(),
            "euclidean_loss": euclidean_loss.detach().item(),
            "x0_loss": x0_loss.detach().item(),
            "decode_loss": decode_loss.detach().item(),
            "weighted_decode_loss": (DECODE_LOSS_WEIGHT * decode_loss).detach().item(),
            "metric_mean": g_diag.detach().mean().item(),
            "metric_std": g_diag.detach().std().item(),
            "metric_min": g_diag.detach().min().item(),
            "metric_max": g_diag.detach().max().item(),
            "metric_reg": metric_reg.detach().item(),
            "metric_reg_mult": float(metric_reg_mult),
            "rollout_loss": rollout_loss.detach().item(),
            "rollout_decode_loss": rollout_decode_loss.detach().item(),
            "weighted_rollout_decode_loss": (ROLLOUT_DECODE_LOSS_WEIGHT * rollout_decode_loss).detach().item(),
            "rollout_hidden_loss": rollout_hidden_loss.detach().item(),
            "weighted_rollout_hidden_loss": (ROLLOUT_HIDDEN_LOSS_WEIGHT * rollout_hidden_loss).detach().item(),
            "rollout_logit_kl": rollout_logit_kl.detach().item(),
            "weighted_rollout_logit_kl": (ROLLOUT_LOGIT_KL_WEIGHT * rollout_logit_kl).detach().item(),
            "rollout_entropy_loss": rollout_entropy_loss.detach().item(),
            "weighted_rollout_entropy_loss": ((ROLLOUT_ENTROPY_LOSS_WEIGHT * rollout_entropy_mult) * rollout_entropy_loss).detach().item(),
            "rollout_entropy_mult": float(rollout_entropy_mult),
            "rollout_gen_entropy": rollout_gen_entropy,
            "rollout_oracle_entropy": rollout_oracle_entropy,
            "rollout_gated_gen_ce": rollout_gated_gen_ce.detach().item(),
            "weighted_rollout_gated_gen_ce": (ROLLOUT_GATED_GEN_CE_WEIGHT * rollout_gated_gen_ce).detach().item(),
            "rollout_gated_gen_ce_active": rollout_gated_gen_ce_active,
            "rollout_gated_gen_ce_top1": rollout_gated_gen_ce_top1,
            "rollout_target_prob_loss": rollout_target_prob_loss.detach().item(),
            "weighted_rollout_target_prob_loss": (ROLLOUT_TARGET_PROB_WEIGHT * rollout_target_prob_loss).detach().item(),
            "rollout_target_prob_active": rollout_target_prob_active,
            "rollout_target_prob_gen": rollout_target_prob_gen,
            "rollout_target_prob_oracle": rollout_target_prob_oracle,
            "rollout_logit_balance": rollout_logit_balance.detach().item(),
            "weighted_rollout_logit_balance": (ROLLOUT_LOGIT_BALANCE_WEIGHT * rollout_logit_balance).detach().item(),
            "rollout_logit_topmass": rollout_logit_topmass,
            "rollout_norm_loss": rollout_norm_loss.detach().item(),
            "rollout_diversity_loss": rollout_diversity_loss.detach().item(),
            "weighted_rollout_diversity_loss": (ROLLOUT_DIVERSITY_LOSS_WEIGHT * rollout_diversity_loss).detach().item(),
            "decoder_adapt_real_ce": decoder_adapt_real_ce.detach().item(),
            "weighted_decoder_adapt_real_ce": (DECODER_ADAPT_REAL_CE_WEIGHT * decoder_adapt_real_ce).detach().item(),
            "decoder_adapt_gen_ce": decoder_adapt_gen_ce.detach().item(),
            "decoder_adapt_gen_ce_mult": float(decoder_adapt_gen_ce_mult),
            "weighted_decoder_adapt_gen_ce": ((DECODER_ADAPT_GEN_CE_WEIGHT * decoder_adapt_gen_ce_mult) * decoder_adapt_gen_ce).detach().item(),
            "decoder_adapt_preserve_kl": decoder_adapt_preserve_kl.detach().item(),
            "weighted_decoder_adapt_preserve_kl": (DECODER_ADAPT_PRESERVE_KL_WEIGHT * decoder_adapt_preserve_kl).detach().item(),
            "self_gate": self_gate,
            "cross_gate": cross_gate,
            "gate_reg": gate_reg.detach().item(),
        }
    return total_loss


def natural_velocity(flow_net, metric_net, z, t, z_cond, pos):
    v = flow_net(z, t, z_cond, pos)
    pooled_cond = z_cond.mean(dim=1).unsqueeze(1).expand_as(z)
    g = metric_net(
        z.reshape(-1, z.size(-1)),
        t.reshape(-1),
        pooled_cond.reshape(-1, z.size(-1)),
        pos.reshape(-1),
    ).reshape_as(z)
    return v / g.clamp_min(1e-3), g


def generate_suffix(flow_net, metric_net, z_cond, batch_size, suffix_len, latent_dim, device, steps=ODE_STEPS, mask=None):
    pos = suffix_positions(batch_size, suffix_len, device)
    z = torch.randn(batch_size, suffix_len, latent_dim, device=device) * BASE_NOISE_STD
    z_initial = z.clone()
    dt = 1.0 / steps
    metric_snapshot = None
    for i in range(steps):
        t = torch.full((batch_size, suffix_len), i / steps, device=device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            v, metric_snapshot = natural_velocity(flow_net, metric_net, z, t, z_cond, pos)
        z = z + v * dt
        if mask is not None:
            z = z * mask.to(z.dtype).unsqueeze(-1)
    z_uncalibrated = z.clone()
    z = calibrate_latents(z, mask)
    return z, metric_snapshot, z_initial, z_uncalibrated


def generated_decode_stats(z_real, z_gen_suffix, suffix_mask, input_ids, attn_mask, decoder, device):
    z_gen_flat = z_gen_suffix[suffix_mask]
    real_flat = z_real[:, PROMPT_LEN:, :][suffix_mask]
    gen_mean = z_gen_flat.mean().item()
    gen_std = z_gen_flat.std().item()
    cosine_sim = F.cosine_similarity(
        real_flat.mean(0, keepdim=True),
        z_gen_flat.mean(0, keepdim=True),
    ).item()

    decode_idx = (attn_mask[:, PROMPT_LEN:].sum(dim=1) > 0).nonzero(as_tuple=False).flatten()
    decode_idx = decode_idx[:DECODE_LOSS_BATCH]
    if decode_idx.numel() == 0:
        return gen_mean, gen_std, cosine_sim, 0.0

    z_decode_gen = torch.cat([z_real[:, :PROMPT_LEN, :], z_gen_suffix], dim=1)[decode_idx]
    decode_targets = input_ids[decode_idx, PROMPT_LEN:].reshape(-1)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        gen_decode_logits = decoder.decode_from_latent(z_decode_gen)
    gen_decode_ce = F.cross_entropy(
        gen_decode_logits[:, PROMPT_LEN:, :].reshape(-1, gen_decode_logits.size(-1)),
        decode_targets,
        ignore_index=0,
    ).item()
    return gen_mean, gen_std, cosine_sim, gen_decode_ce


def argmax_token_collapse_stats(logits, token_ids, tokenizer):
    suffix_logits = logits[:, PROMPT_LEN:, :].float()
    suffix_ids = token_ids[:, PROMPT_LEN:]
    probs = suffix_logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean().item()

    unique_ratios = []
    max_fracs = []
    flat_tokens = []
    for row in suffix_ids:
        valid = row[~torch.isin(row, row.new_tensor(tokenizer.all_special_ids))]
        if valid.numel() == 0:
            continue
        counts = torch.bincount(valid.cpu(), minlength=logits.size(-1))
        unique_ratios.append((counts > 0).sum().item() / valid.numel())
        max_fracs.append(counts.max().item() / valid.numel())
        flat_tokens.append(valid.cpu())

    if flat_tokens:
        flat = torch.cat(flat_tokens)
        counts = torch.bincount(flat, minlength=logits.size(-1))
        top_counts, top_ids = counts.topk(min(5, counts.numel()))
        top_tokens = [
            f"{tokenizer.convert_ids_to_tokens(int(token_id))}:{int(count)}"
            for token_id, count in zip(top_ids.tolist(), top_counts.tolist())
            if count > 0
        ]
    else:
        top_tokens = []

    return {
        "entropy": entropy,
        "unique_ratio": sum(unique_ratios) / max(len(unique_ratios), 1),
        "max_frac": sum(max_fracs) / max(len(max_fracs), 1),
        "top_tokens": top_tokens,
    }


def decoder_distribution_stats(logits, tokenizer, mask=None, oracle_logits=None, target_ids=None):
    suffix_logits = logits[:, PROMPT_LEN:, :].float()
    probs = suffix_logits.softmax(dim=-1)
    top1_acc = 0.0
    target_prob_mean = 0.0
    oracle_top1_acc = 0.0
    oracle_target_prob_mean = 0.0

    if target_ids is not None:
        suffix_targets = target_ids[:, PROMPT_LEN:] if target_ids.size(1) == logits.size(1) else target_ids
        target_valid = suffix_targets != 0
        if mask is not None:
            target_valid = target_valid & mask.bool()
        if target_valid.any():
            target_gather_ids = suffix_targets.clamp(0, probs.size(-1) - 1).unsqueeze(-1)
            target_probs = probs.gather(dim=-1, index=target_gather_ids).squeeze(-1)
            top_ids = probs.argmax(dim=-1)
            top1_acc = (top_ids[target_valid] == suffix_targets[target_valid]).float().mean().item()
            target_prob_mean = target_probs[target_valid].mean().item()

            if oracle_logits is not None:
                oracle_probs_full = oracle_logits[:, PROMPT_LEN:, :].float().softmax(dim=-1)
                oracle_target_probs = oracle_probs_full.gather(dim=-1, index=target_gather_ids).squeeze(-1)
                oracle_top_ids = oracle_probs_full.argmax(dim=-1)
                oracle_top1_acc = (
                    oracle_top_ids[target_valid] == suffix_targets[target_valid]
                ).float().mean().item()
                oracle_target_prob_mean = oracle_target_probs[target_valid].mean().item()

    if mask is not None:
        valid = mask.bool()
        if valid.any():
            probs = probs[valid]
            suffix_logits = suffix_logits[valid]
            if oracle_logits is not None:
                oracle_suffix_logits = oracle_logits[:, PROMPT_LEN:, :].float()[valid]
        else:
            probs = probs.reshape(-1, probs.size(-1))
            suffix_logits = suffix_logits.reshape(-1, suffix_logits.size(-1))
            oracle_suffix_logits = None
    else:
        probs = probs.reshape(-1, probs.size(-1))
        suffix_logits = suffix_logits.reshape(-1, suffix_logits.size(-1))
        oracle_suffix_logits = (
            oracle_logits[:, PROMPT_LEN:, :].float().reshape(-1, oracle_logits.size(-1))
            if oracle_logits is not None
            else None
        )

    special_ids = torch.tensor(tokenizer.all_special_ids, device=probs.device)
    entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
    top_probs, _ = probs.topk(min(50, probs.size(-1)), dim=-1)
    mean_probs = probs.clone()
    mean_probs[:, special_ids] = 0.0
    mean_probs = mean_probs.mean(dim=0)
    batch_top_mass = mean_probs.topk(min(8, mean_probs.numel())).values.sum()
    special_mass = probs[:, special_ids].sum(dim=-1)

    kl_to_oracle = None
    if oracle_logits is not None and oracle_suffix_logits is not None:
        oracle_log_probs = F.log_softmax(oracle_suffix_logits, dim=-1)
        gen_log_probs = F.log_softmax(suffix_logits, dim=-1)
        kl_to_oracle = F.kl_div(gen_log_probs, oracle_log_probs.exp(), reduction="batchmean").item()

    return {
        "entropy": entropy.mean().item(),
        "top1_prob": top_probs[:, 0].mean().item(),
        "top5_mass": top_probs[:, :min(5, top_probs.size(1))].sum(dim=-1).mean().item(),
        "top50_mass": top_probs.sum(dim=-1).mean().item(),
        "batch_top8_mass": batch_top_mass.item(),
        "special_mass": special_mass.mean().item(),
        "kl_to_oracle": kl_to_oracle,
        "top1_acc": top1_acc,
        "target_prob": target_prob_mean,
        "oracle_top1_acc": oracle_top1_acc,
        "oracle_target_prob": oracle_target_prob_mean,
    }


def evaluate(flow_net, metric_net, encoder, decoder, tokenizer, val_loader, device, n_samples=4):
    flow_net.eval()
    metric_net.eval()
    encoder.eval()
    decoder.eval()

    val_loss = 0
    eval_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                z_data = decoder.compress(encoder(input_ids, attention_mask))
                z_cond = prompt_condition(z_data, attention_mask)
                z_target = z_data[:, PROMPT_LEN:, :]
                target_mask = attention_mask[:, PROMPT_LEN:]
                val_loss += flow_matching_loss(flow_net, metric_net, z_target, z_cond, target_mask).item()
    avg_val_loss = val_loss / len(val_loader)

    with torch.no_grad():
        seed_everything(SEED + 10_000)
        batch = next(iter(val_loader))
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        z_real = decoder.compress(encoder(input_ids, attn_mask))
        B, S, D = z_real.shape
        suffix_mask = attn_mask[:, PROMPT_LEN:].bool()
        z_real_suffix = z_real[:, PROMPT_LEN:, :]
        z_real_flat = z_real_suffix[suffix_mask]
        z_cond = prompt_condition(z_real, attn_mask)
        z_gen_suffix, metric_snapshot, z_initial_suffix, z_uncalibrated_suffix = generate_suffix(
            flow_net,
            metric_net,
            z_cond,
            B,
            S - PROMPT_LEN,
            D,
            device,
            mask=suffix_mask,
        )
        z_gen_flat = z_gen_suffix[suffix_mask]
        z_initial_flat = z_initial_suffix[suffix_mask]
        z_uncalibrated_flat = z_uncalibrated_suffix[suffix_mask]

        real_mean = z_real_flat.mean().item()
        real_std = z_real_flat.std().item()
        real_norm = z_real_flat.norm(dim=-1).mean().item()
        initial_mean = z_initial_flat.mean().item()
        initial_std = z_initial_flat.std().item()
        initial_norm = z_initial_flat.norm(dim=-1).mean().item()
        uncal_mean = z_uncalibrated_flat.mean().item()
        uncal_std = z_uncalibrated_flat.std().item()
        uncal_norm = z_uncalibrated_flat.norm(dim=-1).mean().item()
        gen_mean, gen_std, cosine_sim, gen_decode_ce = generated_decode_stats(
            z_real,
            z_gen_suffix,
            suffix_mask,
            input_ids,
            attn_mask,
            decoder,
            device,
        )
        _, _, _, initial_decode_ce = generated_decode_stats(
            z_real,
            z_initial_suffix,
            suffix_mask,
            input_ids,
            attn_mask,
            decoder,
            device,
        )
        _, _, _, uncal_decode_ce = generated_decode_stats(
            z_real,
            z_uncalibrated_suffix,
            suffix_mask,
            input_ids,
            attn_mask,
            decoder,
            device,
        )
        metric_valid = metric_snapshot[suffix_mask] if metric_snapshot is not None else z_gen_flat.new_ones(z_gen_flat.shape)
        metric_mean = metric_valid.mean().item()
        metric_std = metric_valid.std().item()
        metric_min = metric_valid.min().item()
        metric_max = metric_valid.max().item()

        decode_idx = (attn_mask[:, PROMPT_LEN:].sum(dim=1) > 0).nonzero(as_tuple=False).flatten()
        decode_idx = decode_idx[:DECODE_LOSS_BATCH]
        if decode_idx.numel() > 0:
            z_decode_real = z_real[decode_idx]
            decode_targets = input_ids[decode_idx, PROMPT_LEN:].reshape(-1)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                real_decode_logits = decoder.decode_from_latent(z_decode_real)
            real_decode_ce = F.cross_entropy(
                real_decode_logits[:, PROMPT_LEN:, :].reshape(-1, real_decode_logits.size(-1)),
                decode_targets,
                ignore_index=0,
            ).item()
        else:
            real_decode_ce = 0.0
        decode_ce_gap = gen_decode_ce - real_decode_ce

    with torch.no_grad():
        sample_idx = (attn_mask[:, PROMPT_LEN:].sum(dim=1) > 0).nonzero(as_tuple=False).flatten()
        sample_idx = sample_idx[:n_samples]
        z_gen_seq = torch.cat([z_real[:, :PROMPT_LEN, :], z_gen_suffix], dim=1)[sample_idx]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = decoder.decode_from_latent(z_gen_seq)
            oracle_logits = decoder.decode_from_latent(z_real[sample_idx])
        pred_ids = logits.argmax(-1)
        sampled_ids = sample_token_ids(logits, tokenizer)
        oracle_ids = oracle_logits.argmax(-1)
        gen_collapse = argmax_token_collapse_stats(logits, pred_ids, tokenizer)
        sampled_collapse = argmax_token_collapse_stats(logits, sampled_ids, tokenizer)
        oracle_collapse = argmax_token_collapse_stats(oracle_logits, oracle_ids, tokenizer)
        sample_suffix_mask = attn_mask[sample_idx, PROMPT_LEN:]
        sample_target_ids = input_ids[sample_idx]
        gen_dist = decoder_distribution_stats(
            logits,
            tokenizer,
            sample_suffix_mask,
            oracle_logits=oracle_logits,
            target_ids=sample_target_ids,
        )
        oracle_dist = decoder_distribution_stats(
            oracle_logits,
            tokenizer,
            sample_suffix_mask,
            target_ids=sample_target_ids,
        )
        print("\n-- riemannian samples -----------------------------------------")
        for sample_pos, batch_idx in enumerate(sample_idx.tolist()):
            prompt = tokenizer.decode(input_ids[batch_idx, :PROMPT_LEN], skip_special_tokens=True)
            target = tokenizer.decode(input_ids[batch_idx, PROMPT_LEN:], skip_special_tokens=True)
            sampled = tokenizer.decode(sampled_ids[sample_pos, PROMPT_LEN:], skip_special_tokens=True)
            argmax = tokenizer.decode(pred_ids[sample_pos, PROMPT_LEN:], skip_special_tokens=True)
            oracle = tokenizer.decode(oracle_ids[sample_pos, PROMPT_LEN:], skip_special_tokens=True)
            print(f"  prompt:     {prompt}")
            print(f"  target:     {target[:120]}")
            print(f"  oracle:     {oracle[:120]}")
            print(f"  generated:  {sampled[:120]}")
            print(f"  argmax:     {argmax[:120]}")
            print()
        print(
            "  collapse argmax: "
            f"entropy={gen_collapse['entropy']:.2f} "
            f"uniq={gen_collapse['unique_ratio']:.3f} "
            f"maxfrac={gen_collapse['max_frac']:.3f} "
            f"top={', '.join(gen_collapse['top_tokens'])}"
        )
        print(
            "  collapse gen   : "
            f"entropy={sampled_collapse['entropy']:.2f} "
            f"uniq={sampled_collapse['unique_ratio']:.3f} "
            f"maxfrac={sampled_collapse['max_frac']:.3f} "
            f"top={', '.join(sampled_collapse['top_tokens'])}"
        )
        print(
            "  collapse oracle: "
            f"entropy={oracle_collapse['entropy']:.2f} "
            f"uniq={oracle_collapse['unique_ratio']:.3f} "
            f"maxfrac={oracle_collapse['max_frac']:.3f} "
            f"top={', '.join(oracle_collapse['top_tokens'])}"
        )
        print(
            "  dist gen      : "
            f"ent={gen_dist['entropy']:.2f} "
            f"top1={gen_dist['top1_prob']:.3f} "
            f"top5={gen_dist['top5_mass']:.3f} "
            f"top50={gen_dist['top50_mass']:.3f} "
            f"batch_top8={gen_dist['batch_top8_mass']:.3f} "
            f"special={gen_dist['special_mass']:.3f} "
            f"oracle_to_gen_kl={gen_dist['kl_to_oracle']:.3f}"
        )
        print(
            "  target gen    : "
            f"top1_acc={gen_dist['top1_acc']:.3f} "
            f"target_prob={gen_dist['target_prob']:.4f} "
            f"oracle_top1_acc={gen_dist['oracle_top1_acc']:.3f} "
            f"oracle_target_prob={gen_dist['oracle_target_prob']:.4f}"
        )
        print(
            "  dist oracle   : "
            f"ent={oracle_dist['entropy']:.2f} "
            f"top1={oracle_dist['top1_prob']:.3f} "
            f"top5={oracle_dist['top5_mass']:.3f} "
            f"top50={oracle_dist['top50_mass']:.3f} "
            f"batch_top8={oracle_dist['batch_top8_mass']:.3f} "
            f"special={oracle_dist['special_mass']:.3f} "
            f"target_prob={oracle_dist['target_prob']:.4f} "
            f"top1_acc={oracle_dist['top1_acc']:.3f}"
        )
        print()

    torch.random.set_rng_state(eval_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)

    latent_std_gap = abs(gen_std - real_std)
    raw_norm_gap = abs(uncal_norm - real_norm)
    collapse_uniq_penalty = max(0.0, COLLAPSE_UNIQ_TARGET - gen_collapse["unique_ratio"])
    collapse_maxfrac_penalty = max(0.0, gen_collapse["max_frac"] - COLLAPSE_MAXFRAC_TARGET)
    val_score = (
        avg_val_loss
        + latent_std_gap
        + max(0.0, 0.8 - cosine_sim)
        + 0.05 * max(0.0, decode_ce_gap)
        + RAW_NORM_GAP_SCORE_WEIGHT * raw_norm_gap
        + COLLAPSE_UNIQ_SCORE_WEIGHT * collapse_uniq_penalty
        + COLLAPSE_MAXFRAC_SCORE_WEIGHT * collapse_maxfrac_penalty
    )

    print("-- val metrics ------------------------------------------------")
    print(f"  val loss     : {avg_val_loss:.4f}")
    print(f"  real latents : mean={real_mean:.3f}  std={real_std:.3f}  norm={real_norm:.3f}")
    print(f"  gen latents  : mean={gen_mean:.3f}  std={gen_std:.3f}")
    print(f"  init latents : mean={initial_mean:.3f}  std={initial_std:.3f}  norm={initial_norm:.3f}")
    print(f"  raw flow lat : mean={uncal_mean:.3f}  std={uncal_std:.3f}  norm={uncal_norm:.3f}  norm_gap={raw_norm_gap:.3f}")
    print(f"  metric diag  : mean={metric_mean:.3f}  std={metric_std:.3f}  min={metric_min:.3f}  max={metric_max:.3f}")
    print(f"  cosine sim   : {cosine_sim:.4f}")
    print(f"  decoder CE   : real={real_decode_ce:.4f}  init={initial_decode_ce:.4f}  raw={uncal_decode_ce:.4f}  gen={gen_decode_ce:.4f}  gap={decode_ce_gap:.4f}")
    print(f"  collapse pen : uniq={collapse_uniq_penalty:.4f}  maxfrac={collapse_maxfrac_penalty:.4f}")
    print(f"  ode steps    : {ODE_STEPS}")
    print(f"  val score    : {val_score:.4f}")
    print()

    return avg_val_loss, val_score


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

EPOCHS = 20

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
                f" | rdloss {stats['rollout_decode_loss']:.4f}*{ROLLOUT_DECODE_LOSS_WEIGHT:.3f}={stats['weighted_rollout_decode_loss']:.4f}"
                f" | rhid {stats['rollout_hidden_loss']:.4f}*{ROLLOUT_HIDDEN_LOSS_WEIGHT:.3f}={stats['weighted_rollout_hidden_loss']:.4f}"
                f" | oracle2gen_kl {stats['rollout_logit_kl']:.4f}*{ROLLOUT_LOGIT_KL_WEIGHT:.3f}={stats['weighted_rollout_logit_kl']:.4f}"
                f" | entgap {stats['rollout_entropy_loss']:.4f}*{ROLLOUT_ENTROPY_LOSS_WEIGHT:.3f}={stats['weighted_rollout_entropy_loss']:.4f}"
                f" ent g/o={stats['rollout_gen_entropy']:.2f}/{stats['rollout_oracle_entropy']:.2f}"
                f" ent_mult={stats['rollout_entropy_mult']:.2f}"
                f" | rgce {stats['rollout_gated_gen_ce']:.4f}*{ROLLOUT_GATED_GEN_CE_WEIGHT:.3f}={stats['weighted_rollout_gated_gen_ce']:.4f}"
                f" act={stats['rollout_gated_gen_ce_active']:.2f}"
                f" top1={stats['rollout_gated_gen_ce_top1']:.3f}"
                f" | rtp {stats['rollout_target_prob_loss']:.4f}*{ROLLOUT_TARGET_PROB_WEIGHT:.3f}={stats['weighted_rollout_target_prob_loss']:.4f}"
                f" act={stats['rollout_target_prob_active']:.2f}"
                f" p={stats['rollout_target_prob_gen']:.3f}/{stats['rollout_target_prob_oracle']:.3f}"
                f" | rbal {stats['rollout_logit_balance']:.4f}*{ROLLOUT_LOGIT_BALANCE_WEIGHT:.3f}={stats['weighted_rollout_logit_balance']:.4f}"
                f" topm={stats['rollout_logit_topmass']:.3f}"
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
            "rollout_decode_loss_weight": ROLLOUT_DECODE_LOSS_WEIGHT,
            "rollout_hidden_loss_weight": ROLLOUT_HIDDEN_LOSS_WEIGHT,
            "rollout_logit_kl_weight": ROLLOUT_LOGIT_KL_WEIGHT,
            "rollout_logit_kl_temp": ROLLOUT_LOGIT_KL_TEMP,
            "rollout_logit_kl_direction": "oracle_to_gen",
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
            "rollout_logit_balance_weight": ROLLOUT_LOGIT_BALANCE_WEIGHT,
            "rollout_logit_balance_topk": ROLLOUT_LOGIT_BALANCE_TOPK,
            "rollout_logit_balance_target": ROLLOUT_LOGIT_BALANCE_TARGET,
            "logit_balance_special_ids": LOGIT_BALANCE_SPECIAL_IDS,
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
