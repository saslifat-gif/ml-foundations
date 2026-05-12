import argparse
import hashlib
import os
import re
import sys

import torch
from datasets import DownloadConfig, load_dataset
from transformers import BertForMaskedLM, BertTokenizer

sys.path.insert(0, ".")
from parallel_decoder import BertEncoder, ParallelDecoder, cached_from_pretrained
from stage2_riemannian import AuxTokenHead, DenoisingPrior, DenoisingPriorSampler, DraftPriorSampler, FlowNet, MetricNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPT_LEN = 16
MAX_SEQ_LEN = 64
BASE_NOISE_STD = 0.30
CALIBRATE_GENERATED_LATENTS = True
TARGET_LATENT_MEAN = -0.003
TARGET_LATENT_STD = 0.280
DECODE_TEMPERATURE = 0.8
DECODE_TOP_K = 50
DECODE_TOP_P = 0.95
FLOW_HIDDEN_DIM = 512
FLOW_DEPTH = 5
FLOW_REFINE_SCALE = 0.01
VELOCITY_CLAMP = 2.0
METRIC_HIDDEN_DIM = 256
METRIC_LOG_BOUND = 0.75
ODE_STEPS = 16
SELF_GATE_SCALE = 0.10
CROSS_GATE_SCALE = 0.10
GATE_INIT = 0.20
CHAIN_ALPHAS = [0.3, 0.5, 0.7]   # Path A fallback: chain from pure noise
DRAFT_ALPHA = 0.7                  # Path B: draft-encode conditioning alpha (also test 0.5)
MLM_DRAFT_LEN = MAX_SEQ_LEN - PROMPT_LEN
RETRIEVAL_DRAFT_LIMIT = 20000
_RETRIEVAL_SENTENCES = None
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with", "you", "your",
    "tell", "me", "about", "explain",
}


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
def draft_keywords(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def template_draft(user_text):
    clean = " ".join(re.findall(r"[a-zA-Z0-9]+", user_text.lower()))
    if not clean:
        clean = "the topic"
    return f"this is about {clean} and gives a simple explanation with important details and examples"


def split_sentences(text):
    text = " ".join(text.strip().split())
    if len(text) < 30:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if 40 <= len(p.strip()) <= 220]


def load_retrieval_sentences():
    global _RETRIEVAL_SENTENCES
    if _RETRIEVAL_SENTENCES is not None:
        return _RETRIEVAL_SENTENCES
    try:
        ds = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split="validation",
            download_config=DownloadConfig(local_files_only=True),
        )
    except Exception as exc:
        print(f"retrieval draft unavailable from local WikiText cache: {exc}", flush=True)
        _RETRIEVAL_SENTENCES = []
        return _RETRIEVAL_SENTENCES

    sentences = []
    for row in ds:
        for sentence in split_sentences(row.get("text", "")):
            sentences.append(sentence)
            if len(sentences) >= RETRIEVAL_DRAFT_LIMIT:
                _RETRIEVAL_SENTENCES = sentences
                return _RETRIEVAL_SENTENCES
    _RETRIEVAL_SENTENCES = sentences
    print(f"loaded {len(sentences)} retrieval draft sentences", flush=True)
    return _RETRIEVAL_SENTENCES


def retrieval_draft(user_text):
    query = draft_keywords(user_text)
    if not query:
        return template_draft(user_text), "template"
    best_sentence = None
    best_score = 0.0
    for sentence in load_retrieval_sentences():
        sent_terms = draft_keywords(sentence)
        if not sent_terms:
            continue
        overlap = len(query & sent_terms)
        if overlap == 0:
            continue
        score = overlap / (len(query) ** 0.5 * len(sent_terms) ** 0.25)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    if best_sentence is None:
        return template_draft(user_text), "template"
    return best_sentence, "retrieval"


def make_chatbot_draft(user_text, source):
    if source == "template":
        return template_draft(user_text), "template"
    return retrieval_draft(user_text)


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


def encode_suffix_latents(encoder, decoder, full_ids, full_mask):
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        hidden = encoder(full_ids, full_mask)
    z_full = decoder.compress(hidden)
    return z_full[:, PROMPT_LEN:, :]


def generate_latent_draft_tokens(encoder, decoder, z_prompt_exp, prompt_ids, attention_mask,
                                 suffix_len, n_samples, latent_dim, device, z_init=None):
    """Path B: decode z_init → draft tokens → re-encode → on-manifold z_draft.
    z_init should be the chain prior output, not pure noise, for useful instance structure."""
    if z_init is None:
        z_init = torch.randn(n_samples, suffix_len, latent_dim, device=device) * BASE_NOISE_STD
    z_seq = torch.cat([z_prompt_exp, z_init], dim=1)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = decoder.decode_from_latent(z_seq)
    draft_suffix_ids = logits[:, PROMPT_LEN:, :].argmax(dim=-1)  # (n, suffix_len)

    prompt_ids_exp = prompt_ids.expand(n_samples, -1)
    full_ids = torch.cat([prompt_ids_exp, draft_suffix_ids], dim=1)
    draft_mask = torch.ones(n_samples, suffix_len, device=device, dtype=attention_mask.dtype)
    full_mask = torch.cat([attention_mask.expand(n_samples, -1), draft_mask], dim=1)

    return encode_suffix_latents(encoder, decoder, full_ids, full_mask)


def generate_mlm_draft_tokens(mlm_model, tokenizer, prompt_ids, attention_mask, suffix_len, n_samples, device):
    prompt_ids_exp = prompt_ids.expand(n_samples, -1)
    prompt_mask_exp = attention_mask.expand(n_samples, -1)
    suffix_ids = torch.full(
        (n_samples, suffix_len),
        tokenizer.mask_token_id,
        device=device,
        dtype=prompt_ids.dtype,
    )
    suffix_mask = torch.ones(n_samples, suffix_len, device=device, dtype=attention_mask.dtype)
    full_ids = torch.cat([prompt_ids_exp, suffix_ids], dim=1)
    full_mask = torch.cat([prompt_mask_exp, suffix_mask], dim=1)

    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = mlm_model(input_ids=full_ids, attention_mask=full_mask).logits
    draft_suffix_logits = logits[:, PROMPT_LEN:, :]
    draft_suffix_ids = sample_token_ids(
        draft_suffix_logits,
        tokenizer,
        temperature=DECODE_TEMPERATURE,
        top_k=DECODE_TOP_K,
        top_p=DECODE_TOP_P,
    )
    full_ids = torch.cat([prompt_ids_exp, draft_suffix_ids], dim=1)
    return full_ids, full_mask


def generate_mlm_draft_latents(mlm_model, tokenizer, encoder, decoder, prompt_ids, attention_mask,
                               suffix_len, n_samples, device):
    full_ids, full_mask = generate_mlm_draft_tokens(
        mlm_model,
        tokenizer,
        prompt_ids,
        attention_mask,
        suffix_len,
        n_samples,
        device,
    )
    z_draft = encode_suffix_latents(encoder, decoder, full_ids, full_mask)
    draft_texts = [
        tokenizer.decode(full_ids[i, PROMPT_LEN:], skip_special_tokens=True)
        for i in range(n_samples)
    ]
    return z_draft, draft_texts


def generate_manual_draft_latents(tokenizer, encoder, decoder, prompt_text, draft_text, seq_len, n_samples, device):
    full_draft = draft_text
    if prompt_text.lower() not in draft_text.lower():
        full_draft = f"{prompt_text} {draft_text}"
    inputs = tokenizer(
        full_draft,
        return_tensors="pt",
        max_length=seq_len,
        padding="max_length",
        truncation=True,
    )
    full_ids = inputs["input_ids"].to(device).expand(n_samples, -1)
    full_mask = inputs["attention_mask"].to(device).expand(n_samples, -1)
    z_draft = encode_suffix_latents(encoder, decoder, full_ids, full_mask)
    draft_suffix = tokenizer.decode(full_ids[0, PROMPT_LEN:], skip_special_tokens=True)
    return z_draft, [draft_suffix for _ in range(n_samples)]


def denoise_draft_latents(start_prior, z_draft, z_cond, suffix_len, device):
    if start_prior is None or not hasattr(start_prior, "prior"):
        return z_draft
    pos = suffix_positions(z_draft.size(0), suffix_len, device)
    beta = (1.0 - DRAFT_ALPHA ** 2) ** 0.5
    z_t = DRAFT_ALPHA * z_draft + beta * (
        torch.randn_like(z_draft) * TARGET_LATENT_STD + TARGET_LATENT_MEAN
    )
    alpha_t = z_cond.new_full((z_draft.size(0),), DRAFT_ALPHA)
    return start_prior.prior(z_t, z_cond, alpha_t, pos)


def natural_velocity(flow_net, metric_net, z, t, z_cond, pos):
    v = flow_net(z, t, z_cond, pos)
    pooled_cond = z_cond.mean(dim=1).unsqueeze(1).expand_as(z)
    g = metric_net(
        z.reshape(-1, z.size(-1)),
        t.reshape(-1),
        pooled_cond.reshape(-1, z.size(-1)),
        pos.reshape(-1),
    ).reshape_as(z)
    v_nat = v / g.clamp_min(1e-3)
    if VELOCITY_CLAMP is not None and VELOCITY_CLAMP > 0:
        v_nat = VELOCITY_CLAMP * torch.tanh(v_nat / VELOCITY_CLAMP)
    return v_nat, g


def sample_suffix_latents(
    flow_net,
    metric_net,
    z_cond,
    n_samples,
    suffix_len,
    latent_dim,
    device,
    steps=ODE_STEPS,
    start_prior=None,
    z_draft=None,
    z_start=None,
):
    pos = suffix_positions(n_samples, suffix_len, device)
    if z_start is not None:
        z = z_start
    elif z_draft is not None and start_prior is not None and hasattr(start_prior, "prior"):
        # Path B: draft-conditioned prior — z_t = alpha*z_draft + beta*noise → prior(z_t)
        z = denoise_draft_latents(start_prior, z_draft, z_cond, suffix_len, device)
    elif start_prior is not None and hasattr(start_prior, "prior"):
        # Path A fallback: chain mode — pure noise → prior(0.3) → prior(0.5) → prior(0.7)
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
        z = z + FLOW_REFINE_SCALE * v * dt
    z = calibrate_latents(z)
    return z, metric_snapshot


def decode_suffix_texts(decoder, tokenizer, z_prompt_exp, z_suffix, temperature=DECODE_TEMPERATURE, top_k=DECODE_TOP_K, top_p=DECODE_TOP_P):
    z_seq = torch.cat([z_prompt_exp, z_suffix], dim=1)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = decoder.decode_from_latent(z_seq)
    pred_ids = sample_token_ids(logits, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p)
    return [tokenizer.decode(pred_ids[i, PROMPT_LEN:], skip_special_tokens=True) for i in range(z_suffix.size(0))]


def decode_fused_suffix_texts(
    decoder,
    tokenizer,
    flow_net,
    aux_token_head,
    fusion_beta,
    z_prompt_exp,
    z_suffix,
    temperature=DECODE_TEMPERATURE,
    top_k=DECODE_TOP_K,
    top_p=DECODE_TOP_P,
):
    if aux_token_head is None or fusion_beta <= 0:
        return None
    n_samples, suffix_len, _ = z_suffix.shape
    pos = suffix_positions(n_samples, suffix_len, z_suffix.device)
    t = torch.ones((n_samples, suffix_len), device=z_suffix.device, dtype=z_suffix.dtype)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        z_seq = torch.cat([z_prompt_exp, z_suffix], dim=1)
        logits = decoder.decode_from_latent(z_seq).float()
        _v, hidden = flow_net(z_suffix, t, z_prompt_exp, pos, return_hidden=True)
        aux_logits = aux_token_head(hidden).float()
    logits[:, PROMPT_LEN:, :] = logits[:, PROMPT_LEN:, :] + fusion_beta * aux_logits
    pred_ids = sample_token_ids(logits, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p)
    return [tokenizer.decode(pred_ids[i, PROMPT_LEN:], skip_special_tokens=True) for i in range(z_suffix.size(0))]


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
    global FLOW_REFINE_SCALE, VELOCITY_CLAMP
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
    FLOW_REFINE_SCALE = ckpt2.get("flow_refine_scale", FLOW_REFINE_SCALE)
    VELOCITY_CLAMP = ckpt2.get("velocity_clamp", VELOCITY_CLAMP)
    print(f"flow_refine_scale={FLOW_REFINE_SCALE:.3f} velocity_clamp={VELOCITY_CLAMP:.3f}", flush=True)
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
    aux_token_head = None
    aux_logit_fusion_beta = ckpt2.get("aux_logit_fusion_beta", 0.0)
    if ckpt2.get("draft_prior") or ckpt2.get("denoising_prior"):
        is_draft_prior = bool(ckpt2.get("draft_prior"))
        dp_path = ckpt2.get("draft_prior_path") or ckpt2.get("denoising_prior_path", "denoising_prior_best.pt")
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
            if is_draft_prior:
                start_prior = DraftPriorSampler(_dp, latent_dim=256, alpha=dp_alpha).to(device)
            else:
                start_prior = DenoisingPriorSampler(_dp, latent_dim=256, alpha=dp_alpha).to(device)
            start_prior.eval()
            print(
                f"loaded {'draft' if is_draft_prior else 'denoising'} prior from {dp_path} "
                f"(alpha={dp_alpha:.2f})",
                flush=True,
            )
        else:
            print(f"WARNING: prior path not found: {dp_path}", flush=True)

    if ckpt2.get("aux_token_head") is not None:
        aux_token_head = AuxTokenHead(
            hidden_dim=ckpt2.get("aux_token_hidden_dim", FLOW_HIDDEN_DIM),
            vocab_size=tokenizer.vocab_size,
        ).to(device)
        aux_token_head.load_state_dict(ckpt2["aux_token_head"], strict=False)
        aux_token_head.eval()
        print(f"loaded aux fusion head (beta={aux_logit_fusion_beta:.3f})", flush=True)

    encoder.eval()
    decoder.eval()
    flow_net.eval()
    metric_net.eval()
    mlm_model = None
    print(f"loaded {stage1_path} + {stage2_path}")
    return encoder, decoder, flow_net, metric_net, start_prior, aux_token_head, aux_logit_fusion_beta, mlm_model


def load_mlm_model():
    mlm_model = cached_from_pretrained(BertForMaskedLM).to(device)
    mlm_model.eval()
    print("loaded BERT MLM draft model", flush=True)
    return mlm_model


@torch.no_grad()
def generate(
    prompt_text,
    flow_net,
    metric_net,
    encoder,
    decoder,
    mlm_model,
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
    aux_token_head=None,
    aux_logit_fusion_beta=0.0,
    n_refine=0,
    draft_text=None,
    use_mlm_draft=False,
    allow_latent_fallback=False,
    return_debug=False,
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
    suffix_len = min(seq_len - PROMPT_LEN, MLM_DRAFT_LEN)
    z_prompt_exp = z_prompt.expand(n_samples, PROMPT_LEN, latent_dim)

    draft_source = "manual"
    if draft_text:
        z_draft, draft_texts = generate_manual_draft_latents(
            tokenizer,
            encoder,
            decoder,
            prompt_text,
            draft_text,
            seq_len,
            n_samples,
            device,
        )
        z_draft = z_draft[:, :suffix_len, :]
    elif use_mlm_draft and mlm_model is not None:
        draft_source = "mlm"
        z_draft, draft_texts = generate_mlm_draft_latents(
            mlm_model,
            tokenizer,
            encoder,
            decoder,
            input_ids,
            attention_mask,
            suffix_len,
            n_samples,
            device,
        )
    elif not allow_latent_fallback:
        raise ValueError(
            "A rough draft is required for the current working pipeline. "
            "Provide draft_text, use --mlm-draft, or opt into --allow-latent-fallback for diagnostics."
        )
    else:
        draft_source = "latent_chain"
        if start_prior is not None and hasattr(start_prior, "prior"):
            dp = start_prior.prior
            pos_draft = suffix_positions(n_samples, suffix_len, device)
            z_chain = (
                torch.randn(n_samples, suffix_len, latent_dim, device=device)
                * TARGET_LATENT_STD
                + TARGET_LATENT_MEAN
            )
            for alpha_val in CHAIN_ALPHAS:
                alpha_t = z_cond.new_full((n_samples,), alpha_val)
                z_chain = dp(z_chain, z_cond, alpha_t, pos_draft)
            z_draft = generate_latent_draft_tokens(
                encoder,
                decoder,
                z_prompt_exp,
                input_ids,
                attention_mask,
                suffix_len,
                n_samples,
                latent_dim,
                device,
                z_init=z_chain,
            )
            draft_texts = decode_suffix_texts(decoder, tokenizer, z_prompt_exp, z_draft, temperature=0.0)
        else:
            z_draft = torch.randn(n_samples, suffix_len, latent_dim, device=device) * BASE_NOISE_STD
            draft_texts = ["<gaussian fallback>" for _ in range(n_samples)]
    z_prior = denoise_draft_latents(start_prior, z_draft, z_cond, suffix_len, device)
    prior_texts = decode_suffix_texts(
        decoder,
        tokenizer,
        z_prompt_exp,
        z_prior,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

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
        z_start=z_prior,
    )

    # Iterative refinement: encode ODE output → new draft → prior → ODE, repeated n_refine times.
    # Each pass projects the current best latent onto the encoder manifold (via decode→reencode),
    # giving the prior a stronger instance signal than the original chain output.
    if n_refine > 0 and start_prior is not None and hasattr(start_prior, "prior"):
        for _ in range(n_refine):
            z_draft = generate_latent_draft_tokens(
                encoder, decoder, z_prompt_exp,
                input_ids, attention_mask,
                suffix_len, n_samples, latent_dim, device,
                z_init=z,
            )
            z, metric_snapshot = sample_suffix_latents(
                flow_net, metric_net, z_cond, n_samples, suffix_len, latent_dim, device,
                steps=steps, start_prior=start_prior, z_draft=z_draft,
            )

    texts = decode_suffix_texts(
        decoder,
        tokenizer,
        z_prompt_exp,
        z,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    fused_texts = decode_fused_suffix_texts(
        decoder,
        tokenizer,
        flow_net,
        aux_token_head,
        aux_logit_fusion_beta,
        z_prompt_exp,
        z,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    metric_text = (
        f"metric diag mean={metric_snapshot.mean().item():.3f} std={metric_snapshot.std().item():.3f} "
        f"| draft_alpha={DRAFT_ALPHA:.2f}"
    )
    if return_debug:
        return {
            "draft_source": draft_source,
            "draft": draft_texts,
            "prior": prior_texts,
            "flow": texts,
            "fused": fused_texts,
            "metric": metric_text,
        }
    return texts, metric_text


@torch.no_grad()
def diagnose(
    flow_net,
    metric_net,
    encoder,
    decoder,
    mlm_model,
    tokenizer,
    device,
    steps=ODE_STEPS,
    start_prior=None,
    aux_token_head=None,
    aux_logit_fusion_beta=0.0,
):
    torch.manual_seed(42)

    prompts = [
        "the roman empire was founded",
        "quantum mechanics describes",
        "the amazon rainforest contains",
        "homarus gammarus is a large crustacean",
    ]
    manual_drafts = {
        "the roman empire was founded": (
            "the roman empire was founded after the rise of augustus and became one of the largest powers "
            "in the ancient mediterranean world with armies roads cities and provincial governments"
        ),
        "quantum mechanics describes": (
            "quantum mechanics describes the behaviour of matter and energy at atomic scales where particles "
            "can act like waves and measurements are described by probabilities"
        ),
        "the amazon rainforest contains": (
            "the amazon rainforest contains a vast diversity of plants animals rivers and indigenous communities "
            "and it plays an important role in the climate of south america"
        ),
        "homarus gammarus is a large crustacean": (
            "homarus gammarus is a large crustacean with a hard blue shell powerful claws and a habitat in "
            "cold european coastal waters where it feeds on small marine animals"
        ),
    }

    print("\n-- riemannian prompt-conditioned starts -----------------------")
    for prompt in prompts:
        for n_ref in [0, 1, 2]:
            debug = generate(
                prompt,
                flow_net,
                metric_net,
                encoder,
                decoder,
                mlm_model,
                tokenizer,
                n_samples=1,
                steps=steps,
                temperature=DECODE_TEMPERATURE,
                top_k=DECODE_TOP_K,
                top_p=DECODE_TOP_P,
                device=device,
                start_prior=start_prior,
                aux_token_head=aux_token_head,
                aux_logit_fusion_beta=aux_logit_fusion_beta,
                n_refine=n_ref,
                draft_text=manual_drafts[prompt],
                return_debug=True,
            )
            print(f"  prompt:    {prompt}  [refine={n_ref}]")
            print(f"  draft({debug['draft_source']}): {debug['draft'][0][:100]}")
            print(f"  prior:     {debug['prior'][0][:100]}")
            print(f"  flow:      {debug['flow'][0][:100]}")
            if debug["fused"] is not None:
                print(f"  fused:     {debug['fused'][0][:100]}")
            print(f"  {debug['metric']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt-conditioned Riemannian stage2 inference")
    parser.add_argument("--stage1", default="stage1_best.pt", help="path to the stage1 checkpoint")
    parser.add_argument("--stage2", default=None, help="path to the stage2 checkpoint")
    parser.add_argument("--mlm-draft", action="store_true", help="use raw BERT MLM draft when no manual draft is provided")
    parser.add_argument("--allow-latent-fallback", action="store_true", help="allow the weak latent-chain draft fallback for diagnostics")
    parser.add_argument("--chatbot", action="store_true", help="auto-create a rough draft, then refine it")
    parser.add_argument("--draft-source", choices=("retrieval", "template"), default="retrieval", help="draft source for --chatbot blank drafts")
    args = parser.parse_args()

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, flow_net, metric_net, start_prior, aux_token_head, aux_logit_fusion_beta, mlm_model = load_models(args.stage1, args.stage2)
    if args.mlm_draft:
        mlm_model = load_mlm_model()

    diagnose(
        flow_net,
        metric_net,
        encoder,
        decoder,
        mlm_model,
        tokenizer,
        device,
        start_prior=start_prior,
        aux_token_head=aux_token_head,
        aux_logit_fusion_beta=aux_logit_fusion_beta,
    )

    mode = "draft-then-refine chatbot" if args.chatbot else "interactive refinement"
    print(f"\n{mode} mode - press Ctrl+C to exit\n")
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
            n_ref = int(input("refine passes? [default 0]: ") or 0)
            draft = input("rough draft? [blank=auto in chatbot mode]: ").strip()
            if not draft and args.chatbot:
                draft, draft_source = make_chatbot_draft(prompt, args.draft_source)
                print(f"auto draft({draft_source}): {draft}\n")
            if not draft and not args.mlm_draft and not args.allow_latent_fallback:
                print("rough draft is required for the working refinement path; blank latent fallback is disabled\n")
                continue
            debug = generate(
                prompt,
                flow_net,
                metric_net,
                encoder,
                decoder,
                mlm_model,
                tokenizer,
                n_samples=n,
                steps=s,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                start_prior=start_prior,
                aux_token_head=aux_token_head,
                aux_logit_fusion_beta=aux_logit_fusion_beta,
                n_refine=n_ref,
                draft_text=draft or None,
                use_mlm_draft=args.mlm_draft,
                allow_latent_fallback=args.allow_latent_fallback,
                return_debug=True,
            )
            print(debug["metric"])
            print()
            for i in range(n):
                print(f"  [{i+1}] draft({debug['draft_source']}): {debug['draft'][i]}\n")
                print(f"      prior:     {debug['prior'][i]}\n")
                print(f"      flow:      {debug['flow'][i]}\n")
                if debug["fused"] is not None:
                    print(f"      fused:     {debug['fused'][i]}\n")
        except KeyboardInterrupt:
            break
