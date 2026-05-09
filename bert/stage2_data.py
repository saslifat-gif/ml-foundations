import random

import torch
from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader

from stage2_config import DATALOADER_NUM_WORKERS, SEED


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_stage2_dataloaders(tokenizer, train_size, batch_size, max_length):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    try:
        ds = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            download_config=DownloadConfig(local_files_only=True),
        )
        print("loaded wikitext from local datasets cache", flush=True)
    except Exception as exc:
        print(f"local wikitext cache unavailable ({exc}) | trying online load", flush=True)
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_size = min(train_size, len(ds["train"]))
    small_train = ds["train"].select(range(train_size))
    small_val = ds["validation"]

    small_train = small_train.filter(lambda x: len(x["text"].strip()) > 10)
    small_val = small_val.filter(lambda x: len(x["text"].strip()) > 10)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    train_tok = small_train.map(tokenize, batched=True)
    val_tok = small_val.map(tokenize, batched=True)
    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(
        train_tok,
        batch_size=batch_size,
        shuffle=True,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=DATALOADER_NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_tok,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=DATALOADER_NUM_WORKERS > 0,
    )
    print(
        f"train batches: {len(train_loader)}  val batches: {len(val_loader)}  "
        f"max_length: {max_length}",
        flush=True,
    )
    return train_loader, val_loader
