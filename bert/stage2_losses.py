import torch
import torch.nn.functional as F

from stage2_config import *
from stage2_riemannian import (
    attention_gate_regularizer,
    attention_gate_stats,
    natural_velocity,
    suffix_positions,
)


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


def rollout_cosine_alignment_loss(z_pred, z_target, mask=None):
    token_cos = F.cosine_similarity(z_pred.float(), z_target.detach().float(), dim=-1)
    if mask is not None:
        valid = mask.bool()
        if not valid.any():
            return z_pred.new_tensor(0.0), 0.0
        token_cos = token_cos[valid]
    mean_cos = token_cos.mean()
    return 1.0 - mean_cos, mean_cos.detach().item()


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


def rollout_flow_token_ce_loss(gen_logits, suffix_ids, mask=None):
    suffix_logits = gen_logits[:, PROMPT_LEN:, :].float()
    suffix_targets = suffix_ids[:gen_logits.size(0)]
    valid = suffix_targets != 0
    if mask is not None:
        valid = valid & mask.bool()
    if not valid.any():
        return gen_logits.new_tensor(0.0), 0.0, 0.0

    token_ce = F.cross_entropy(
        suffix_logits.reshape(-1, suffix_logits.size(-1)),
        suffix_targets.reshape(-1),
        ignore_index=0,
        reduction="none",
    ).reshape(valid.shape)
    probs = suffix_logits.softmax(dim=-1)
    target_gather_ids = suffix_targets.clamp(0, probs.size(-1) - 1).unsqueeze(-1)
    target_prob = probs.gather(dim=-1, index=target_gather_ids).squeeze(-1)
    top1 = probs.max(dim=-1).values
    return (
        token_ce[valid].mean(),
        target_prob[valid].detach().mean().item(),
        top1[valid].detach().mean().item(),
    )


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
                    "rollout_entropy_loss": 0.0,
                    "weighted_rollout_entropy_loss": 0.0,
                    "rollout_entropy_mult": 0.0,
                    "rollout_gen_entropy": 0.0,
                    "rollout_oracle_entropy": 0.0,
                    "rollout_gated_gen_ce": 0.0,
                    "weighted_rollout_gated_gen_ce": 0.0,
                    "rollout_gated_gen_ce_active": 0.0,
                    "rollout_gated_gen_ce_top1": 0.0,
                    "rollout_flow_token_ce": 0.0,
                    "weighted_rollout_flow_token_ce": 0.0,
                    "rollout_flow_token_ce_target_prob": 0.0,
                    "rollout_flow_token_ce_top1": 0.0,
                    "rollout_target_prob_loss": 0.0,
                    "weighted_rollout_target_prob_loss": 0.0,
                    "rollout_target_prob_active": 0.0,
                    "rollout_target_prob_gen": 0.0,
                    "rollout_target_prob_oracle": 0.0,
                    "rollout_norm_loss": 0.0,
                    "rollout_diversity_loss": 0.0,
                    "weighted_rollout_diversity_loss": 0.0,
                    "rollout_cosine_loss": 0.0,
                    "weighted_rollout_cosine_loss": 0.0,
                    "rollout_cosine": 0.0,
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
    rollout_entropy_loss = z_target.new_tensor(0.0)
    rollout_entropy_mult = entropy_weight_multiplier(global_step, steps_per_epoch)
    rollout_gen_entropy = 0.0
    rollout_oracle_entropy = 0.0
    rollout_gated_gen_ce = z_target.new_tensor(0.0)
    rollout_gated_gen_ce_active = 0.0
    rollout_gated_gen_ce_top1 = 0.0
    rollout_flow_token_ce = z_target.new_tensor(0.0)
    rollout_flow_token_ce_target_prob = 0.0
    rollout_flow_token_ce_top1 = 0.0
    rollout_target_prob_loss = z_target.new_tensor(0.0)
    rollout_target_prob_active = 0.0
    rollout_target_prob_gen = 0.0
    rollout_target_prob_oracle = 0.0
    rollout_norm_loss = z_target.new_tensor(0.0)
    rollout_diversity_loss = z_target.new_tensor(0.0)
    rollout_cosine_loss = z_target.new_tensor(0.0)
    rollout_cosine = 0.0
    decoder_adapt_real_ce = z_target.new_tensor(0.0)
    decoder_adapt_gen_ce = z_target.new_tensor(0.0)
    decoder_adapt_preserve_kl = z_target.new_tensor(0.0)
    teacher_real_logits = None
    decoder_adapt_gen_ce_mult = 1.0
    if DECODER_ADAPT_GEN_CE_RAMP_EPOCHS > 0 and global_step is not None and steps_per_epoch:
        ramp_steps = max(1, int(DECODER_ADAPT_GEN_CE_RAMP_EPOCHS * steps_per_epoch))
        decoder_adapt_gen_ce_mult = min(1.0, (global_step + 1) / ramp_steps)
    need_rollout = (
        ROLLOUT_TRAIN_STEPS > 0
        and (
            ROLLOUT_LOSS_WEIGHT > 0
            or ROLLOUT_NORM_LOSS_WEIGHT > 0
            or ROLLOUT_DIVERSITY_LOSS_WEIGHT > 0
            or ROLLOUT_COSINE_LOSS_WEIGHT > 0
            or ROLLOUT_FLOW_TOKEN_CE_WEIGHT > 0
            or ROLLOUT_GATED_GEN_CE_WEIGHT > 0
            or ROLLOUT_TARGET_PROB_WEIGHT > 0
            or (ROLLOUT_ENTROPY_LOSS_WEIGHT > 0 and rollout_entropy_mult > 0)
            or DECODER_ADAPT
        )
    )
    if need_rollout:
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
        if ROLLOUT_COSINE_LOSS_WEIGHT > 0:
            rollout_cosine_loss, rollout_cosine = rollout_cosine_alignment_loss(z_roll, z_roll_target, roll_mask)

        need_entropy_logits = (
            decoder is not None
            and z_prompt is not None
            and (
                (ROLLOUT_ENTROPY_LOSS_WEIGHT > 0 and rollout_entropy_mult > 0)
                or (ROLLOUT_GATED_GEN_CE_WEIGHT > 0 and suffix_ids is not None)
                or (ROLLOUT_FLOW_TOKEN_CE_WEIGHT > 0 and suffix_ids is not None)
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
            if entropy_decoder is teacher_decoder:
                teacher_real_logits = oracle_entropy_logits
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
            if ROLLOUT_FLOW_TOKEN_CE_WEIGHT > 0 and suffix_ids is not None:
                n_flow_token_ce = n_rollout
                if ROLLOUT_FLOW_TOKEN_CE_BATCH > 0:
                    n_flow_token_ce = min(n_flow_token_ce, ROLLOUT_FLOW_TOKEN_CE_BATCH)
                flow_token_mask = roll_mask[:n_flow_token_ce] if roll_mask is not None else None
                (
                    rollout_flow_token_ce,
                    rollout_flow_token_ce_target_prob,
                    rollout_flow_token_ce_top1,
                ) = rollout_flow_token_ce_loss(
                    gen_entropy_logits[:n_flow_token_ce],
                    suffix_ids[:n_flow_token_ce],
                    flow_token_mask,
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

            if DECODER_ADAPT_PRESERVE_KL_WEIGHT > 0 and teacher_decoder is not None:
                if real_logits is None:
                    real_logits = decoder.decode_from_latent(z_real_seq)
                if teacher_real_logits is None:
                    with torch.no_grad():
                        teacher_real_logits = teacher_decoder.decode_from_latent(z_real_seq)
                temp = DECODER_ADAPT_KL_TEMP
                token_kl = F.kl_div(
                    F.log_softmax(real_logits[:, PROMPT_LEN:, :].float() / temp, dim=-1),
                    F.softmax(teacher_real_logits[:, PROMPT_LEN:, :].float() / temp, dim=-1),
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
        + (ROLLOUT_ENTROPY_LOSS_WEIGHT * rollout_entropy_mult) * rollout_entropy_loss
        + ROLLOUT_GATED_GEN_CE_WEIGHT * rollout_gated_gen_ce
        + ROLLOUT_FLOW_TOKEN_CE_WEIGHT * rollout_flow_token_ce
        + ROLLOUT_TARGET_PROB_WEIGHT * rollout_target_prob_loss
        + ROLLOUT_NORM_LOSS_WEIGHT * rollout_norm_loss
        + ROLLOUT_DIVERSITY_LOSS_WEIGHT * rollout_diversity_loss
        + ROLLOUT_COSINE_LOSS_WEIGHT * rollout_cosine_loss
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
            "rollout_entropy_loss": rollout_entropy_loss.detach().item(),
            "weighted_rollout_entropy_loss": ((ROLLOUT_ENTROPY_LOSS_WEIGHT * rollout_entropy_mult) * rollout_entropy_loss).detach().item(),
            "rollout_entropy_mult": float(rollout_entropy_mult),
            "rollout_gen_entropy": rollout_gen_entropy,
            "rollout_oracle_entropy": rollout_oracle_entropy,
            "rollout_gated_gen_ce": rollout_gated_gen_ce.detach().item(),
            "weighted_rollout_gated_gen_ce": (ROLLOUT_GATED_GEN_CE_WEIGHT * rollout_gated_gen_ce).detach().item(),
            "rollout_gated_gen_ce_active": rollout_gated_gen_ce_active,
            "rollout_gated_gen_ce_top1": rollout_gated_gen_ce_top1,
            "rollout_flow_token_ce": rollout_flow_token_ce.detach().item(),
            "weighted_rollout_flow_token_ce": (ROLLOUT_FLOW_TOKEN_CE_WEIGHT * rollout_flow_token_ce).detach().item(),
            "rollout_flow_token_ce_target_prob": rollout_flow_token_ce_target_prob,
            "rollout_flow_token_ce_top1": rollout_flow_token_ce_top1,
            "rollout_target_prob_loss": rollout_target_prob_loss.detach().item(),
            "weighted_rollout_target_prob_loss": (ROLLOUT_TARGET_PROB_WEIGHT * rollout_target_prob_loss).detach().item(),
            "rollout_target_prob_active": rollout_target_prob_active,
            "rollout_target_prob_gen": rollout_target_prob_gen,
            "rollout_target_prob_oracle": rollout_target_prob_oracle,
            "rollout_norm_loss": rollout_norm_loss.detach().item(),
            "rollout_diversity_loss": rollout_diversity_loss.detach().item(),
            "weighted_rollout_diversity_loss": (ROLLOUT_DIVERSITY_LOSS_WEIGHT * rollout_diversity_loss).detach().item(),
            "rollout_cosine_loss": rollout_cosine_loss.detach().item(),
            "weighted_rollout_cosine_loss": (ROLLOUT_COSINE_LOSS_WEIGHT * rollout_cosine_loss).detach().item(),
            "rollout_cosine": rollout_cosine,
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
