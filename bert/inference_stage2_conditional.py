import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

sys.path.insert(0, ".")
from parallel_decoder import BertEncoder, ParallelDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPT_LEN = 16


class FlowNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=2048, depth=8):
        super().__init__()
        layers = [nn.Linear(latent_dim * 2 + 1, hidden_dim), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, t, z_cond):
        inp = torch.cat([z_t, z_cond, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)


def prompt_condition(z_prompt, attention_mask):
    prompt_mask = attention_mask[:, :PROMPT_LEN].to(z_prompt.dtype).unsqueeze(-1)
    denom = prompt_mask.sum(dim=1).clamp_min(1.0)
    return (z_prompt[:, :PROMPT_LEN, :] * prompt_mask).sum(dim=1) / denom


def load_models(stage1_path="stage1_best.pt", stage2_path="stage2_conditional_best.pt"):
    encoder  = BertEncoder().to(device)
    decoder  = ParallelDecoder(latent_dim=256).to(device)
    flow_net = FlowNet(latent_dim=256, hidden_dim=2048, depth=8).to(device)

    ckpt1 = torch.load(stage1_path, map_location=device, weights_only=False)
    decoder.load_state_dict(ckpt1["decoder"])
    if "encoder" in ckpt1:
        encoder.load_state_dict(ckpt1["encoder"])

    ckpt2 = torch.load(stage2_path, map_location=device, weights_only=False)
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
             n_samples=4, seq_len=128, latent_dim=256, steps=100, device=device):
    inputs = tokenizer(prompt_text, return_tensors="pt",
                       max_length=PROMPT_LEN, padding="max_length", truncation=True)
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    hidden   = encoder(input_ids, attention_mask)
    z_prompt = decoder.compress(hidden)
    z_cond   = prompt_condition(z_prompt, attention_mask)
    suffix_len = seq_len - PROMPT_LEN
    z_cond = z_cond.expand(n_samples * suffix_len, -1)

    z  = torch.randn(n_samples * suffix_len, latent_dim, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n_samples * suffix_len,), i / steps, device=device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            v = flow_net(z, t, z_cond)
        z = z + v * dt

    z_prompt = z_prompt.expand(n_samples, PROMPT_LEN, latent_dim)
    z_suffix = z.view(n_samples, suffix_len, latent_dim)
    z_seq    = torch.cat([z_prompt, z_suffix], dim=1)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = decoder.decode_from_latent(z_seq)
    pred_ids = logits.argmax(-1)
    return [tokenizer.decode(pred_ids[i], skip_special_tokens=True)
            for i in range(n_samples)]


@torch.no_grad()
def diagnose(flow_net, encoder, decoder, tokenizer, device):
    torch.manual_seed(42)
    noise = torch.randn(128, 256, device=device)

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
        z_cond   = prompt_condition(z_prompt, attention_mask).expand(128 - PROMPT_LEN, -1)

        z  = noise[:128 - PROMPT_LEN].clone()
        dt = 1.0 / 100
        for i in range(100):
            t = torch.full((128 - PROMPT_LEN,), i / 100, device=device)
            with torch.no_grad():
                v_cond   = flow_net(z, t, z_cond)
                v_uncond = flow_net(z, t, torch.zeros_like(z_cond))
                v        = v_uncond + 2.0 * (v_cond - v_uncond)
            z = z + v * dt

        z_seq = torch.cat([z_prompt, z.unsqueeze(0)], dim=1)
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
