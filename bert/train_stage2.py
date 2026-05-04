import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertTokenizer

sys.path.insert(0, ".")
from paralla_decoder import BertEncoder, ParallelDecoder, build_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using: {device}")


class FlowNet(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, z_t, t):
        inp = torch.cat([z_t, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)


# ── load stage 1 ──────────────────────────────────────────────────────────────

encoder = BertEncoder().to(device)
decoder = ParallelDecoder(latent_dim=256).to(device)

checkpoint = torch.load("stage1_best.pt", map_location=device)
decoder.load_state_dict(checkpoint["decoder"])
for param in decoder.parameters():
    param.requires_grad = False
encoder.eval()
print("stage1 loaded and frozen")

# ── data ──────────────────────────────────────────────────────────────────────

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_loader, val_loader = build_dataloaders(tokenizer, train_size=200000)

# ── stage 2 training ──────────────────────────────────────────────────────────

flow_net  = FlowNet(latent_dim=256).to(device)
optimizer = AdamW(flow_net.parameters(), lr=1e-4)

EPOCHS    = 20
best_loss = float("inf")

for epoch in range(EPOCHS):
    flow_net.train()

    train_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            hidden = encoder(input_ids, attention_mask)  # [B, 128, 768]
            z_data = decoder.compress(hidden)             # [B, 128, 256]

        B, S, D = z_data.shape
        z_flat  = z_data.view(B * S, D)
        z_noise = torch.randn_like(z_flat)

        t   = torch.rand(B * S, device=device)
        z_t = (1 - t.unsqueeze(-1)) * z_noise + t.unsqueeze(-1) * z_flat

        v_true = z_flat - z_noise
        v_pred = flow_net(z_t, t)

        loss = F.mse_loss(v_pred, v_true)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_net.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

        if step % 50 == 0:
            print(f"epoch {epoch+1} step {step}/{len(train_loader)} | loss {loss.item():.4f}",
                  flush=True)

    avg_loss = train_loss / len(train_loader)
    print(f"\nepoch {epoch+1} done | avg loss {avg_loss:.4f}\n", flush=True)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({"flow_net": flow_net.state_dict()}, "stage2_best.pt")
        print(f"saved best model at loss {best_loss:.4f}", flush=True)
