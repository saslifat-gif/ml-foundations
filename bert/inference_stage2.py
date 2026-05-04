import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

sys.path.insert(0, ".")
from paralla_decoder import BertEncoder, ParallelDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── FlowNet (must match training arch) ────────────────────────────────────────

class FlowNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=2048, depth=8):
        super().__init__()
        layers = [nn.Linear(latent_dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, t):
        inp = torch.cat([z_t, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)


# ── Load models 1───────────────────────────────────────────────────────────────

def load_models(stage1_path="stage1_best.pt", stage2_path="stage2_best.pt"):
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

# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    flow_net,
    decoder,
    tokenizer,
    n_samples   = 4,
    seq_len     = 128,
    latent_dim  = 256,
    steps       = 100,       # euler integration steps — more = better quality
    device      = device,
):
    # sample noise
    z = torch.randn(n_samples * seq_len, latent_dim, device=device)

    # euler integration: noise → latent
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n_samples * seq_len,), i / steps, device=device)
        with torch.amp.autocast("cuda"):
            v = flow_net(z, t)
        z = z + v * dt

    # decode latents → token ids
    z_seq  = z.view(n_samples, seq_len, latent_dim)
    with torch.amp.autocast("cuda"):
        logits = decoder.decode_from_latent(z_seq)   # [n, seq, vocab]
    pred_ids = logits.argmax(-1)                      # [n, seq]

    texts = [tokenizer.decode(pred_ids[i], skip_special_tokens=True)
             for i in range(n_samples)]
    return texts


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder, decoder, flow_net = load_models()

    print("\ngenerating samples...\n")
    texts = generate(flow_net, decoder, tokenizer, n_samples=4, steps=100)
    for i, text in enumerate(texts):
        print(f"[{i+1}] {text}\n")

    # interactive mode
    print("\ninteractive mode — press Ctrl+C to exit\n")
    while True:
        try:
            n = int(input("how many samples? [default 4]: ") or 4)
            s = int(input("steps? [default 100]: ") or 100)
            texts = generate(flow_net, decoder, tokenizer, n_samples=n, steps=s)
            print()
            for i, text in enumerate(texts):
                print(f"[{i+1}] {text}\n")
        except KeyboardInterrupt:
            break