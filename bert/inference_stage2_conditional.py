import os
import sys

import torch
import torch.nn as nn
from transformers import BertTokenizer

sys.path.insert(0, ".")
from parallel_decoder import BertEncoder, ParallelDecoder, cached_from_pretrained

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPT_LEN = 16
MAX_SEQ_LEN = 32
BASE_NOISE_STD = 0.30
CALIBRATE_GENERATED_LATENTS = True
TARGET_LATENT_MEAN = -0.003
TARGET_LATENT_STD = 0.264
DECODE_TEMPERATURE = 0.9
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

    def forward(self, z_t, t, z_cond, pos):
        inp = torch.cat([z_t, z_cond, t.unsqueeze(-1), pos.unsqueeze(-1)], dim=-1)
        log_g = self.net(inp)
        log_g = log_g - log_g.mean(dim=-1, keepdim=True)
        log_g = self.log_bound * torch.tanh(log_g / self.log_bound)
        g_diag = torch.exp(log_g)
        return g_diag / g_diag.mean(dim=-1, keepdim=True).clamp_min(1e-6)


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


def sample_suffix_latents(flow_net, metric_net, z_cond, n_samples, suffix_len, latent_dim, device, steps=ODE_STEPS):
    pos = suffix_positions(n_samples, suffix_len, device)
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
    decoder.load_state_dict(ckpt1["decoder"])
    if "encoder" in ckpt1:
        encoder.load_state_dict(ckpt1["encoder"])

    ckpt2 = torch.load(stage2_path, map_location=device, weights_only=False)
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

    encoder.eval()
    decoder.eval()
    flow_net.eval()
    metric_net.eval()
    print(f"loaded {stage1_path} + {stage2_path}")
    return encoder, decoder, flow_net, metric_net


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
def diagnose(flow_net, metric_net, encoder, decoder, tokenizer, device, steps=ODE_STEPS):
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
        )
        print(f"  prompt:    {prompt}")
        print(f"  generated: {texts[0][:100]}")
        print(f"  {metric_text}")
        print()


if __name__ == "__main__":
    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, flow_net, metric_net = load_models()

    diagnose(flow_net, metric_net, encoder, decoder, tokenizer, device)

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
            )
            print(metric_text)
            print()
            for i, text in enumerate(texts):
                print(f"  [{i+1}] {text}\n")
        except KeyboardInterrupt:
            break
