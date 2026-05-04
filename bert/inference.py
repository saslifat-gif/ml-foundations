import argparse
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer


# ── Models (must match training architecture) ─────────────────────────────────

class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state   # [B, seq_len, 768]


class ParallelDecoder(nn.Module):
    def __init__(self, latent_dim=256, vocab_size=30522):
        super().__init__()
        self.compress   = nn.Linear(768, latent_dim)
        self.project_up = nn.Linear(latent_dim, 768)
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.is_decoder = False
        self.bert      = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.to_logits = nn.Linear(768, vocab_size)

    def forward(self, z):
        h      = self.compress(z)
        x      = self.project_up(h) + z                # residual
        out    = self.bert(inputs_embeds=x)
        return self.to_logits(out.last_hidden_state)


# ── Load & predict ────────────────────────────────────────────────────────────

def load_models(checkpoint_path, device, latent_dim=256):
    encoder = BertEncoder().to(device)
    decoder = ParallelDecoder(latent_dim=latent_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()
    print(f"loaded checkpoint: {checkpoint_path}")
    return encoder, decoder


def predict(text, encoder, decoder, tokenizer, max_length=128):
    device = next(encoder.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                       padding="max_length", truncation=True)
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        z        = encoder(input_ids, attention_mask)
        logits   = decoder(z)
        pred_ids = logits.argmax(-1)

    # trim to non-padding positions only
    mask      = inputs["attention_mask"][0].bool()
    original  = tokenizer.decode(input_ids[0][mask],      skip_special_tokens=True)
    predicted = tokenizer.decode(pred_ids[0].cpu()[mask],  skip_special_tokens=True)
    return original, predicted


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="stage1_best.pt")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder, decoder = load_models(args.checkpoint, device, args.latent_dim)

    if args.text:
        original, predicted = predict(args.text, encoder, decoder, tokenizer)
        print(f"original:  {original}")
        print(f"predicted: {predicted}")
    else:
        print("Enter text to reconstruct (Ctrl+C to exit)\n")
        while True:
            try:
                text = input(">> ")
                if not text.strip():
                    continue
                original, predicted = predict(text, encoder, decoder, tokenizer)
                print(f"original:  {original}")
                print(f"predicted: {predicted}\n")
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
