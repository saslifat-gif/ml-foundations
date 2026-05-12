import random

import torch
import torch.nn.functional as F

from stage2_config import *
from stage2_losses import flow_matching_loss
from stage2_riemannian import generate_suffix, prompt_condition, suffix_positions


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def latent_mix_decode_ce(z_real, z_gen_suffix, input_ids, attn_mask, decoder, device, alphas=(0.01, 0.03, 0.05, 0.10, 0.20, 0.50)):
    decode_idx = (attn_mask[:, PROMPT_LEN:].sum(dim=1) > 0).nonzero(as_tuple=False).flatten()
    decode_idx = decode_idx[:DECODE_LOSS_BATCH]
    if decode_idx.numel() == 0:
        return []
    z_real_suffix = z_real[:, PROMPT_LEN:, :]
    decode_targets = input_ids[decode_idx, PROMPT_LEN:].reshape(-1)
    results = []
    for alpha in alphas:
        z_mix_suffix = (1.0 - alpha) * z_gen_suffix + alpha * z_real_suffix
        z_mix_seq = torch.cat([z_real[:, :PROMPT_LEN, :], z_mix_suffix], dim=1)[decode_idx]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            mix_logits = decoder.decode_from_latent(z_mix_seq)
        mix_ce = F.cross_entropy(
            mix_logits[:, PROMPT_LEN:, :].reshape(-1, mix_logits.size(-1)),
            decode_targets,
            ignore_index=0,
        ).item()
        results.append((alpha, mix_ce))
    return results


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


def fuse_aux_logits(flow_net, aux_token_head, decoder_logits, z_suffix, z_cond, mask=None):
    if aux_token_head is None or AUX_LOGIT_FUSION_BETA <= 0:
        return decoder_logits
    B, T, _ = z_suffix.shape
    pos = suffix_positions(B, T, z_suffix.device, z_suffix.dtype)
    t = torch.ones(B, T, device=z_suffix.device, dtype=z_suffix.dtype)
    _, aux_hidden = flow_net(z_suffix, t, z_cond, pos, mask, return_hidden=True)
    aux_logits = aux_token_head(aux_hidden).float()
    fused_logits = decoder_logits.float().clone()
    fused_logits[:, PROMPT_LEN:, :] = fused_logits[:, PROMPT_LEN:, :] + AUX_LOGIT_FUSION_BETA * aux_logits
    return fused_logits


def evaluate(
    flow_net,
    metric_net,
    encoder,
    decoder,
    tokenizer,
    val_loader,
    device,
    n_samples=4,
    start_mlp=None,
    aux_token_head=None,
    latent_projector=None,
    draft_start_fn=None,
):
    flow_net.eval()
    metric_net.eval()
    encoder.eval()
    decoder.eval()
    if start_mlp is not None:
        start_mlp.eval()
    if aux_token_head is not None:
        aux_token_head.eval()
    if latent_projector is not None:
        latent_projector.eval()

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
                z_draft_start = draft_start_fn(input_ids, attention_mask) if draft_start_fn is not None else None
                val_loss += flow_matching_loss(
                    flow_net,
                    metric_net,
                    z_target,
                    z_cond,
                    target_mask,
                    start_mlp=start_mlp,
                    z_draft_start=z_draft_start,
                ).item()
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
        z_draft_start = draft_start_fn(input_ids, attn_mask) if draft_start_fn is not None else None
        z_real_flat = z_real_suffix[suffix_mask]
        z_cond = prompt_condition(z_real, attn_mask)
        if z_draft_start is not None and start_mlp is not None and hasattr(start_mlp, "set_draft_target"):
            start_mlp.set_draft_target(z_draft_start)
        z_gen_suffix, metric_snapshot, z_initial_suffix, z_uncalibrated_suffix = generate_suffix(
            flow_net,
            metric_net,
            z_cond,
            B,
            S - PROMPT_LEN,
            D,
            device,
            mask=suffix_mask,
            start_mlp=start_mlp,
            z_target_start=z_real_suffix if STRUCTURED_TARGET_START else None,
        )
        z_projected_suffix = z_gen_suffix
        if latent_projector is not None:
            z_projected_suffix, projector_delta = latent_projector(
                z_gen_suffix,
                z_real[:, :PROMPT_LEN, :],
                suffix_mask,
            )
            projector_delta_norm = projector_delta[suffix_mask].norm(dim=-1).mean().item()
        else:
            projector_delta_norm = 0.0
        z_gen_flat = z_gen_suffix[suffix_mask]
        z_projected_flat = z_projected_suffix[suffix_mask]
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
            z_projected_suffix,
            suffix_mask,
            input_ids,
            attn_mask,
            decoder,
            device,
        )
        mix_decode_ce = latent_mix_decode_ce(
            z_real,
            z_gen_suffix,
            input_ids,
            attn_mask,
            decoder,
            device,
        )
        raw_gen_mean = z_gen_flat.mean().item()
        raw_gen_std = z_gen_flat.std().item()
        projected_norm = z_projected_flat.norm(dim=-1).mean().item()
        if start_mlp is not None:
            pos_start = suffix_positions(B, S - PROMPT_LEN, device, z_real.dtype)
            if z_draft_start is not None and hasattr(start_mlp, "set_draft_target"):
                start_mlp.set_draft_target(z_draft_start)
            z_coarse_suffix = start_mlp(z_cond, pos_start, suffix_mask)
            coarse_mean, coarse_std, coarse_cosine, coarse_decode_ce = generated_decode_stats(
                z_real,
                z_coarse_suffix,
                suffix_mask,
                input_ids,
                attn_mask,
                decoder,
                device,
            )
            coarse_mse = F.mse_loss(z_coarse_suffix[suffix_mask], z_real_suffix[suffix_mask]).item()
            coarse_decode_idx = (attn_mask[:, PROMPT_LEN:].sum(dim=1) > 0).nonzero(as_tuple=False).flatten()
            coarse_decode_idx = coarse_decode_idx[:DECODE_LOSS_BATCH]
            if coarse_decode_idx.numel() > 0:
                coarse_seq = torch.cat([z_real[:, :PROMPT_LEN, :], z_coarse_suffix], dim=1)[coarse_decode_idx]
                coarse_targets = input_ids[coarse_decode_idx, PROMPT_LEN:]
                coarse_valid = coarse_targets != 0
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    coarse_logits = decoder.decode_from_latent(coarse_seq)[:, PROMPT_LEN:, :].float()
                coarse_probs = coarse_logits.softmax(dim=-1)
                coarse_target_ids = coarse_targets.clamp(0, coarse_probs.size(-1) - 1).unsqueeze(-1)
                coarse_target_probs = coarse_probs.gather(dim=-1, index=coarse_target_ids).squeeze(-1)
                coarse_target_prob = coarse_target_probs[coarse_valid].mean().item() if coarse_valid.any() else 0.0
            else:
                coarse_target_prob = 0.0
        else:
            coarse_mean = 0.0
            coarse_std = 0.0
            coarse_cosine = 0.0
            coarse_decode_ce = 0.0
            coarse_mse = 0.0
            coarse_target_prob = 0.0
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
            z_decode_gen = torch.cat([z_real[:, :PROMPT_LEN, :], z_projected_suffix], dim=1)[decode_idx]
            decode_targets = input_ids[decode_idx, PROMPT_LEN:].reshape(-1)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                real_decode_logits = decoder.decode_from_latent(z_decode_real)
                gen_decode_logits = decoder.decode_from_latent(z_decode_gen)
                fused_decode_logits = fuse_aux_logits(
                    flow_net,
                    aux_token_head,
                    gen_decode_logits,
                    z_projected_suffix[decode_idx],
                    z_cond[decode_idx],
                    suffix_mask[decode_idx],
                )
            real_decode_ce = F.cross_entropy(
                real_decode_logits[:, PROMPT_LEN:, :].reshape(-1, real_decode_logits.size(-1)),
                decode_targets,
                ignore_index=0,
            ).item()
            fused_decode_ce = F.cross_entropy(
                fused_decode_logits[:, PROMPT_LEN:, :].reshape(-1, fused_decode_logits.size(-1)),
                decode_targets,
                ignore_index=0,
            ).item()
            fused_suffix_logits = fused_decode_logits[:, PROMPT_LEN:, :].float()
            fused_targets = input_ids[decode_idx, PROMPT_LEN:]
            fused_valid = fused_targets != 0
            if fused_valid.any():
                fused_probs = fused_suffix_logits.softmax(dim=-1)
                fused_target_ids = fused_targets.clamp(0, fused_probs.size(-1) - 1).unsqueeze(-1)
                fused_decode_target_prob = (
                    fused_probs.gather(dim=-1, index=fused_target_ids).squeeze(-1)[fused_valid]
                    .mean()
                    .item()
                )
            else:
                fused_decode_target_prob = 0.0
        else:
            real_decode_ce = 0.0
            fused_decode_ce = gen_decode_ce
            fused_decode_target_prob = 0.0
        decode_ce_gap = gen_decode_ce - real_decode_ce
        fused_decode_ce_gap = fused_decode_ce - real_decode_ce

    with torch.no_grad():
        sample_idx = (attn_mask[:, PROMPT_LEN:].sum(dim=1) > 0).nonzero(as_tuple=False).flatten()
        sample_idx = sample_idx[:n_samples]
        z_gen_seq = torch.cat([z_real[:, :PROMPT_LEN, :], z_projected_suffix], dim=1)[sample_idx]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            raw_logits = decoder.decode_from_latent(z_gen_seq)
            logits = fuse_aux_logits(
                flow_net,
                aux_token_head,
                raw_logits,
                z_projected_suffix[sample_idx],
                z_cond[sample_idx],
                attn_mask[sample_idx, PROMPT_LEN:],
            )
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
        raw_gen_dist = decoder_distribution_stats(
            raw_logits,
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
            "  target raw    : "
            f"top1_acc={raw_gen_dist['top1_acc']:.3f} "
            f"target_prob={raw_gen_dist['target_prob']:.4f}"
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
        + 0.05 * max(0.0, fused_decode_ce_gap if aux_token_head is not None else decode_ce_gap)
        + RAW_NORM_GAP_SCORE_WEIGHT * raw_norm_gap
        + COLLAPSE_UNIQ_SCORE_WEIGHT * collapse_uniq_penalty
        + COLLAPSE_MAXFRAC_SCORE_WEIGHT * collapse_maxfrac_penalty
    )

    print("-- val metrics ------------------------------------------------")
    print(f"  val loss     : {avg_val_loss:.4f}")
    print(f"  real latents : mean={real_mean:.3f}  std={real_std:.3f}  norm={real_norm:.3f}")
    print(f"  gen latents  : mean={gen_mean:.3f}  std={gen_std:.3f}")
    if start_mlp is not None:
        print(f"  coarse start : mean={coarse_mean:.3f}  std={coarse_std:.3f}  mse={coarse_mse:.4f}  cos={coarse_cosine:.4f}  ce={coarse_decode_ce:.4f}  p={coarse_target_prob:.4f}")
    if latent_projector is not None:
        print(f"  raw gen lat  : mean={raw_gen_mean:.3f}  std={raw_gen_std:.3f}")
        print(f"  projector    : delta_norm={projector_delta_norm:.3f}  proj_norm={projected_norm:.3f}")
    print(f"  init latents : mean={initial_mean:.3f}  std={initial_std:.3f}  norm={initial_norm:.3f}")
    print(f"  raw flow lat : mean={uncal_mean:.3f}  std={uncal_std:.3f}  norm={uncal_norm:.3f}  norm_gap={raw_norm_gap:.3f}")
    print(f"  metric diag  : mean={metric_mean:.3f}  std={metric_std:.3f}  min={metric_min:.3f}  max={metric_max:.3f}")
    print(f"  cosine sim   : {cosine_sim:.4f}")
    print(f"  decoder CE   : real={real_decode_ce:.4f}  init={initial_decode_ce:.4f}  raw={uncal_decode_ce:.4f}  gen={gen_decode_ce:.4f}  gap={decode_ce_gap:.4f}")
    if aux_token_head is not None:
        print(
            f"  fused CE     : beta={AUX_LOGIT_FUSION_BETA:.3f}  gen={fused_decode_ce:.4f}  "
            f"gap={fused_decode_ce_gap:.4f}  p={fused_decode_target_prob:.4f}"
        )
    if mix_decode_ce:
        mix_text = "  ".join(f"a={alpha:.2f}:{ce:.3f}" for alpha, ce in mix_decode_ce)
        print(f"  mix CE raw   : {mix_text}")
    print(f"  collapse pen : uniq={collapse_uniq_penalty:.4f}  maxfrac={collapse_maxfrac_penalty:.4f}")
    print(f"  ode steps    : {ODE_STEPS}")
    print(f"  val score    : {val_score:.4f}")
    print()

    return avg_val_loss, val_score
