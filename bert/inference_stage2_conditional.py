import argparse
import hashlib
import os
import sys

import torch
from transformers import BertTokenizer

sys.path.insert(0, ".")
from parallel_decoder import BertEncoder, ParallelDecoder, cached_from_pretrained
from stage2_riemannian import DenoisingPrior, DenoisingPriorSampler, FlowNet, MetricNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPT_LEN = 16
MAX_SEQ_LEN = 32
BASE_NOISE_STD = 0.30
CALIBRATE_GENERATED_LATENTS = True
TARGET_LATENT_MEAN = -0.003
TARGET_LATENT_STD = 0.264
DECODE_TEMPERATURE = 0.8
DECODE_TOP_K = 50
DECODE_TOP_P = 0.95
FLOW_HIDDEN_DIM = 512
FLOW_DEPTH = 5
METRIC_HIDDEN_DIM = 256
METRIC_LOG_BOUND = 0.75
ODE_STEPS = 16
SELF_GATE_SCALE = 0.10
CROSS_GATE_SCALE = 0.10
GATE_INIT = 0.20
CHAIN_ALPHAS = [0.3, 0.5, 0.7]   # inference chain order: low → high fidelity


def checkpoint_file_info(path):
    abs_path = os.path.abspath(path)
    stat = os.stat(abs_path)
    return abs_path, stat.st_size, stat.st_mtime


def tensor_state_fingerprint(state_dict, max_tensors=8):
    digest = hashlib.sha256()
    for idx, key in enumerate(sorted(state_dict.keys())):
        if idx >= max_tensors:
            break
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()[:16]


def checkpoint_fingerprint(ckpt):
    parts = []
    if "flow_net" in ckpt:
        parts.append(f"flow={tensor_state_fingerprint(ckpt['flow_net'])}")
    if "metric_net" in ckpt:
        parts.append(f"metric={tensor_state_fingerprint(ckpt['metric_net'])}")
    if "decoder" in ckpt:
        parts.append(f"decoder={tensor_state_fingerprint(ckpt['decoder'])}")
    return " ".join(parts) if parts else "no known model states"


def print_checkpoint_summary(label, path, ckpt):
    abs_path, size_bytes, mtime = checkpoint_file_info(path)
    print(
        f"{label} checkpoint: {abs_path} "
        f"| size={size_bytes / (1024 * 1024):.1f}MB "
        f"| mtime={mtime:.0f}",
        flush=True,
    )
    print(f"{label} fingerprint: {checkpoint_fingerprint(ckpt)}", flush=True)
    if label == "stage2":
        metadata_keys = (
            "stage2_arch",
            "best_score",
            "best_loss",
            "train_size",
            "seed",
            "flow_hidden_dim",
            "flow_depth",
            "metric_hidden_dim",
            "metric_log_bound",
            "decoder_adapt",
            "denoising_prior",
            "denoising_prior_path",
            "denoising_prior_alpha",
            "eval_sample_temperature",
            "eval_sample_top_k",
            "eval_sample_top_p",
        )
        metadata = {key: ckpt[key] for key in metadata_keys if key in ckpt}
        if metadata:
            print(f"stage2 metadata: {metadata}", flush=True)



def prompt_condition(z_prompt, attention_mask):
    prompt_mask = attention_mask[:, :PROMPT_LEN].to(z_prompt.dtype).unsqueeze(-1)
    return z_prompt[:, :PROMPT_LEN, :] * prompt_mask


def suffix_positions(batch_size, suffix_len, device, dtype=torch.float32):
    pos = torch.arange(PROMPT_LEN, PROMPT_LEN + suffix_len, device=device, dtype=dtype)
    pos = pos / max(MAX_SEQ_LEN - 1, 1)
    return pos.unsqueeze(0).expand(batch_size, suffix_len)


def calibrate_latents(z, target_mean=TARGET_LATENT_MEAN, target_std=TARGET_LATENT_STD, eps=1e-6):
    if not CALIBRATE_GENERATED_LATENTS:
        return z
    return (z - z.mean()) * (target_std / z.std().clamp_min(eps)) + target_mean


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


def sample_suffix_latents(flow_net, metric_net, z_cond, n_samples, suffix_len, latent_dim, device, steps=ODE_STEPS, start_prior=None):
    pos = suffix_positions(n_samples, suffix_len, device)
    if start_prior is not None and hasattr(start_prior, "prior"):
        # DenoisingPriorSampler — run inference chain: pure_noise → prior(a) for a in CHAIN_ALPHAS
        dp = start_prior.prior
        z = torch.randn(n_samples, suffix_len, latent_dim, device=device) * TARGET_LATENT_STD + TARGET_LATENT_MEAN
        for alpha_val in CHAIN_ALPHAS:
            alpha_t = z_cond.new_full((n_samples,), alpha_val)
            z = dp(z, z_cond, alpha_t, pos)
    elif start_prior is not None:
        # Other start prior (e.g. StartMLP) — single-step
        z = start_prior(z_cond, pos, mask=None)
    else:
        z = torch.randn(n_samples, suffix_len, latent_dim, device=device) * BASE_NOISE_STD
    dt = 1.0 / steps
    metric_snapshot = None
    for i in range(steps):
        t = torch.full((n_samples, suffix_len), i / steps, device=device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            v, metric_snapshot = natural_velocity(flow_net, metric_net, z, t, z_cond, pos)
        z = z + v * dt
    z = calibrate_latents(z)
    return z, metric_snapshot


def sample_token_ids(logits, tokenizer, temperature=DECODE_TEMPERATURE, top_k=DECODE_TOP_K, top_p=DECODE_TOP_P):
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


def load_models(stage1_path="stage1_best.pt", stage2_path=None):
    if stage2_path is None:
        adapt_path = "stage2_conditional_decoder_adapt_best.pt"
        stage2_path = adapt_path if os.path.exists(adapt_path) else "stage2_conditional_best.pt"

    encoder = BertEncoder().to(device)
    decoder = ParallelDecoder(latent_dim=256).to(device)

    ckpt1 = torch.load(stage1_path, map_location=device, weights_only=False)
    print_checkpoint_summary("stage1", stage1_path, ckpt1)
    decoder.load_state_dict(ckpt1["decoder"])
    if "encoder" in ckpt1:
        encoder.load_state_dict(ckpt1["encoder"])

    ckpt2 = torch.load(stage2_path, map_location=device, weights_only=False)
    print_checkpoint_summary("stage2", stage2_path, ckpt2)
    flow_net = FlowNet(
        latent_dim=256,
        hidden_dim=ckpt2.get("flow_hidden_dim", FLOW_HIDDEN_DIM),
        depth=ckpt2.get("flow_depth", FLOW_DEPTH),
    ).to(device)
    metric_net = MetricNet(
        latent_dim=256,
        hidden_dim=ckpt2.get("metric_hidden_dim", METRIC_HIDDEN_DIM),
        log_bound=ckpt2.get("metric_log_bound", METRIC_LOG_BOUND),
    ).to(device)
    flow_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt2["flow_net"].items()}
    metric_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt2["metric_net"].items()}
    flow_net.load_state_dict(flow_state)
    metric_net.load_state_dict(metric_state)
    if "encoder" in ckpt2:
        encoder.load_state_dict(ckpt2["encoder"])
    if "decoder" in ckpt2:
        decoder.load_state_dict(ckpt2["decoder"])
        print("loaded adapted decoder from stage2 checkpoint", flush=True)

    start_prior = None
    if ckpt2.get("denoising_prior"):
        dp_path = ckpt2.get("denoising_prior_path", "denoising_prior_best.pt")
        dp_alpha = ckpt2.get("denoising_prior_alpha", 0.5)
        if os.path.exists(dp_path):
            dp_ckpt = torch.load(dp_path, map_location=device, weights_only=False)
            _dp = DenoisingPrior(
                latent_dim=256,
                hidden_dim=dp_ckpt.get("denoising_hidden_dim", FLOW_HIDDEN_DIM),
                num_layers=dp_ckpt.get("denoising_layers", 4),
                num_heads=dp_ckpt.get("denoising_heads", 8),
            ).to(device)
            _dp.load_state_dict(dp_ckpt["denoising_prior"])
            _dp.eval()
            start_prior = DenoisingPriorSampler(_dp, latent_dim=256, alpha=dp_alpha).to(device)
            start_prior.eval()
            print(f"loaded denoising prior from {dp_path} (alpha={dp_alpha:.2f})", flush=True)
        else:
            print(f"WARNING: denoising prior path not found: {dp_path}", flush=True)

    encoder.eval()
    decoder.eval()
    flow_net.eval()
    metric_net.eval()
    print(f"loaded {stage1_path} + {stage2_path}")
    return encoder, decoder, flow_net, metric_net, start_prior


@torch.no_grad()
def generate(
    prompt_text,
    flow_net,
    metric_net,
    encoder,
    decoder,
    tokenizer,
    n_samples=4,
    seq_len=MAX_SEQ_LEN,
    latent_dim=256,
    steps=ODE_STEPS,
    temperature=DECODE_TEMPERATURE,
    top_k=DECODE_TOP_K,
    top_p=DECODE_TOP_P,
    device=device,
    start_prior=None,
):
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=PROMPT_LEN,
        padding="max_length",
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    hidden = encoder(input_ids, attention_mask)
    z_prompt = decoder.compress(hidden)
    z_cond = prompt_condition(z_prompt, attention_mask).expand(n_samples, -1, -1)
    suffix_len = seq_len - PROMPT_LEN

    z, metric_snapshot = sample_suffix_latents(
        flow_net,
        metric_net,
        z_cond,
        n_samples,
        suffix_len,
        latent_dim,
        device,
        steps=steps,
        start_prior=start_prior,
    )

    z_prompt = z_prompt.expand(n_samples, PROMPT_LEN, latent_dim)
    z_seq = torch.cat([z_prompt, z], dim=1)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = decoder.decode_from_latent(z_seq)
    pred_ids = sample_token_ids(logits, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p)
    texts = [tokenizer.decode(pred_ids[i], skip_special_tokens=True) for i in range(n_samples)]
    metric_text = f"metric diag mean={metric_snapshot.mean().item():.3f} std={metric_snapshot.std().item():.3f}"
    return texts, metric_text


@torch.no_grad()
def diagnose(flow_net, metric_net, encoder, decoder, tokenizer, device, steps=ODE_STEPS, start_prior=None):
    torch.manual_seed(42)

    prompts = [
        "the roman empire was founded",
        "quantum mechanics describes",
        "the amazon rainforest contains",
        "homarus gammarus is a large crustacean",
    ]

    print("\n-- riemannian prompt-conditioned starts -----------------------")
    for prompt in prompts:
        texts, metric_text = generate(
            prompt,
            flow_net,
            metric_net,
            encoder,
            decoder,
            tokenizer,
            n_samples=1,
            steps=steps,
            temperature=DECODE_TEMPERATURE,
            top_k=DECODE_TOP_K,
            top_p=DECODE_TOP_P,
            device=device,
            start_prior=start_prior,
        )
        print(f"  prompt:    {prompt}")
        print(f"  generated: {texts[0][:100]}")
        print(f"  {metric_text}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt-conditioned Riemannian stage2 inference")
    parser.add_argument("--stage1", default="stage1_best.pt", help="path to the stage1 checkpoint")
    parser.add_argument("--stage2", default=None, help="path to the stage2 checkpoint")
    args = parser.parse_args()

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, flow_net, metric_net, start_prior = load_models(args.stage1, args.stage2)

    diagnose(flow_net, metric_net, encoder, decoder, tokenizer, device, start_prior=start_prior)

    print("\ninteractive mode - press Ctrl+C to exit\n")
    while True:
        try:
            prompt = input("prompt >> ").strip()
            if not prompt:
                continue
            n = int(input("samples? [default 2]: ") or 2)
            s = int(input(f"ode steps? [default {ODE_STEPS}]: ") or ODE_STEPS)
            temp = float(input(f"temperature? [default {DECODE_TEMPERATURE}, 0=argmax]: ") or DECODE_TEMPERATURE)
            top_k = int(input(f"top_k? [default {DECODE_TOP_K}, 0=off]: ") or DECODE_TOP_K)
            top_p = float(input(f"top_p? [default {DECODE_TOP_P}, 0=off]: ") or DECODE_TOP_P)
            top_k = None if top_k <= 0 else top_k
            top_p = None if top_p <= 0 else top_p
            texts, metric_text = generate(
                prompt,
                flow_net,
                metric_net,
                encoder,
                decoder,
                tokenizer,
                n_samples=n,
                steps=s,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                start_prior=start_prior,
            )
            print(metric_text)
            print()
            for i, text in enumerate(texts):
                print(f"  [{i+1}] {text}\n")
        except KeyboardInterrupt:
            break
