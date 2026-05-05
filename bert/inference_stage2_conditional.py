import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

sys.path.insert(0, ".")
from parallel_decoder import BertEncoder, ParallelDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPT_LEN = 16
MAX_SEQ_LEN = 128
BASE_NOISE_STD = 0.30
FLOW_HIDDEN_DIM = 512
FLOW_DEPTH = 4


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
            })
            for _ in range(depth)
        ])
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

        if z_cond.dim() == 3 and z_cond.size(1) == PROMPT_LEN:
            prompt_h = self.prompt_proj(z_cond) + self.prompt_pos
            cond = self.cond_proj(prompt_h.reshape(prompt_h.size(0), -1))
        elif z_cond.dim() == 3:
            cond = z_cond.mean(dim=1)
        else:
            cond = z_cond
        cond = cond.unsqueeze(1).expand(-1, z_t.size(1), -1)

        inp = torch.cat([z_t, cond, t.unsqueeze(-1), pos.unsqueeze(-1)], dim=-1)
        h = self.in_proj(inp)
        if mask is not None:
            h = h * mask.to(h.dtype).unsqueeze(-1)

        for block in self.blocks:
            residual = h
            x = block["norm"](h)
            x = block["conv"](x.transpose(1, 2)).transpose(1, 2)
            x = block["mix"](x)
            h = residual + x
            if mask is not None:
                h = h * mask.to(h.dtype).unsqueeze(-1)

        out = self.out_proj(self.out_norm(h))
        if squeeze:
            out = out.squeeze(1)
        return out


def prompt_condition(z_prompt, attention_mask):
    prompt_mask = attention_mask[:, :PROMPT_LEN].to(z_prompt.dtype).unsqueeze(-1)
    return z_prompt[:, :PROMPT_LEN, :] * prompt_mask


def suffix_positions(batch_size, suffix_len, device, dtype=torch.float32):
    pos = torch.arange(PROMPT_LEN, PROMPT_LEN + suffix_len, device=device, dtype=dtype)
    pos = pos / max(MAX_SEQ_LEN - 1, 1)
    return pos.unsqueeze(0).expand(batch_size, suffix_len)


def load_models(stage1_path="stage1_best.pt", stage2_path="stage2_conditional_best.pt"):
    encoder  = BertEncoder().to(device)
    decoder  = ParallelDecoder(latent_dim=256).to(device)

    ckpt1 = torch.load(stage1_path, map_location=device, weights_only=False)
    decoder.load_state_dict(ckpt1["decoder"])
    if "encoder" in ckpt1:
        encoder.load_state_dict(ckpt1["encoder"])

    ckpt2 = torch.load(stage2_path, map_location=device, weights_only=False)
    flow_hidden_dim = ckpt2.get("flow_hidden_dim", FLOW_HIDDEN_DIM)
    flow_depth = ckpt2.get("flow_depth", FLOW_DEPTH)
    flow_net = FlowNet(latent_dim=256, hidden_dim=flow_hidden_dim, depth=flow_depth).to(device)
    state  = {k.replace("_orig_mod.", ""): v for k, v in ckpt2["flow_net"].items()}
    flow_net.load_state_dict(state)
    if "encoder" in ckpt2:
        encoder.load_state_dict(ckpt2["encoder"])

    encoder.eval()
    decoder.eval()
    flow_net.eval()
    print(f"loaded {stage1_path} + {stage2_path}")
    return encoder, decoder, flow_net


@torch.no_grad()
def generate(prompt_text, flow_net, encoder, decoder, tokenizer,
             n_samples=4, seq_len=128, latent_dim=256, steps=100, guidance_scale=1.0, device=device):
    inputs = tokenizer(prompt_text, return_tensors="pt",
                       max_length=PROMPT_LEN, padding="max_length", truncation=True)
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    hidden   = encoder(input_ids, attention_mask)
    z_prompt = decoder.compress(hidden)
    z_cond   = prompt_condition(z_prompt, attention_mask)
    suffix_len = seq_len - PROMPT_LEN
    z_cond = z_cond.expand(n_samples, -1, -1)
    pos = suffix_positions(n_samples, suffix_len, device)

    z  = torch.randn(n_samples, suffix_len, latent_dim, device=device) * BASE_NOISE_STD
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n_samples, suffix_len), i / steps, device=device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            v = flow_net(z, t, z_cond, pos)
            if guidance_scale != 1.0:
                v_uncond = flow_net(z, t, torch.zeros_like(z_cond), pos)
                v = v_uncond + guidance_scale * (v - v_uncond)
        z = z + v * dt

    z_prompt = z_prompt.expand(n_samples, PROMPT_LEN, latent_dim)
    z_seq    = torch.cat([z_prompt, z], dim=1)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = decoder.decode_from_latent(z_seq)
    pred_ids = logits.argmax(-1)
    return [tokenizer.decode(pred_ids[i], skip_special_tokens=True)
            for i in range(n_samples)]


@torch.no_grad()
def diagnose(flow_net, encoder, decoder, tokenizer, device):
    torch.manual_seed(42)
    noise = torch.randn(128, 256, device=device) * BASE_NOISE_STD

    prompts = [
        "the roman empire was founded",
        "quantum mechanics describes",
        "the amazon rainforest contains",
        "homarus gammarus is a large crustacean",
    ]

    print("\n── same noise, different prompts ─────────────────────────────")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt",
                        max_length=PROMPT_LEN, padding="max_length", truncation=True)
        input_ids      = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        hidden   = encoder(input_ids, attention_mask)
        z_prompt = decoder.compress(hidden)
        z_cond   = prompt_condition(z_prompt, attention_mask)
        pos      = suffix_positions(1, 128 - PROMPT_LEN, device)

        z  = noise[:128 - PROMPT_LEN].unsqueeze(0).clone()
        dt = 1.0 / 100
        for i in range(100):
            t = torch.full((1, 128 - PROMPT_LEN), i / 100, device=device)
            with torch.no_grad():
                v_cond   = flow_net(z, t, z_cond, pos)
                v_uncond = flow_net(z, t, torch.zeros_like(z_cond), pos)
                v        = v_uncond + 2.0 * (v_cond - v_uncond)
            z = z + v * dt

        z_seq = torch.cat([z_prompt, z], dim=1)
        logits   = decoder.decode_from_latent(z_seq)
        pred_ids = logits.argmax(-1)
        print(f"  prompt:    {prompt}")
        print(f"  generated: {tokenizer.decode(pred_ids[0], skip_special_tokens=True)[:100]}")
        print()


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder, decoder, flow_net = load_models()

    diagnose(flow_net, encoder, decoder, tokenizer, device)

    print("\ninteractive mode — press Ctrl+C to exit\n")
    while True:
        try:
            prompt = input("prompt >> ").strip()
            if not prompt:
                continue
            n = int(input("samples? [default 2]: ") or 2)
            s = int(input("steps?   [default 100]: ") or 100)
            texts = generate(prompt, flow_net, encoder, decoder, tokenizer,
                             n_samples=n, steps=s)
            print()
            for i, text in enumerate(texts):
                print(f"  [{i+1}] {text}\n")
        except KeyboardInterrupt:
            break
