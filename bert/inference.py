import argparse
import sys

import torch
from transformers import BertTokenizer

sys.path.insert(0, ".")
from parallel_decoder import BertEncoder, ParallelDecoder


def load_models(checkpoint_path, device, latent_dim=256):
    encoder = BertEncoder().to(device)
    decoder = ParallelDecoder(latent_dim=latent_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    decoder.load_state_dict(checkpoint["decoder"])
    if "encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])

    encoder.eval()
    decoder.eval()
    print(f"loaded checkpoint: {checkpoint_path}")
    return encoder, decoder


def predict(text, encoder, decoder, tokenizer, max_length=128, residual_weight=0.0):
    device = next(encoder.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                       padding="max_length", truncation=True)
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        z        = encoder(input_ids, attention_mask)
        logits   = decoder(z, residual_weight=residual_weight)
        pred_ids = logits.argmax(-1)

    mask      = inputs["attention_mask"][0].bool()
    original  = tokenizer.decode(input_ids[0][mask], skip_special_tokens=True)
    predicted = tokenizer.decode(pred_ids[0].cpu()[mask], skip_special_tokens=True)
    return original, predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="stage1_best.pt")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--residual_weight", type=float, default=0.0)
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder, decoder = load_models(args.checkpoint, device, args.latent_dim)

    if args.text:
        original, predicted = predict(
            args.text, encoder, decoder, tokenizer,
            max_length=args.max_length,
            residual_weight=args.residual_weight,
        )
        print(f"original:  {original}")
        print(f"predicted: {predicted}")
    else:
        print("Enter text to reconstruct (Ctrl+C to exit)\n")
        while True:
            try:
                text = input(">> ")
                if not text.strip():
                    continue
                original, predicted = predict(
                    text, encoder, decoder, tokenizer,
                    max_length=args.max_length,
                    residual_weight=args.residual_weight,
                )
                print(f"original:  {original}")
                print(f"predicted: {predicted}\n")
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
