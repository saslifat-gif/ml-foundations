import argparse
import inspect
import json
import re
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def _norm(v):
    s = str(v).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,:;\"'$%")


def _strip_idx(p):
    return re.sub(r"\[\d+\]", "", p)


def _flatten(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flatten(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(_flatten(v, f"{p}[{i}]"))
    else:
        out[p] = o
    return out


def _bag(o):
    c = Counter()
    for path, val in _flatten(o).items():
        c[(_strip_idx(path), _norm(val))] += 1
    return c


def parse_pred(raw):
    if isinstance(raw, list):
        raw = raw[0].get("content", "") if raw and isinstance(raw[0], dict) else str(raw)
    raw = str(raw).strip()
    if "<|output|>" in raw:
        raw = raw.split("<|output|>", 1)[1]
    if "<|end-output|>" in raw:
        raw = raw.split("<|end-output|>", 1)[0]
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        return json.loads(raw), True
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0)), True
            except Exception:
                return None, False
        return None, False


def field_recall(go, po):
    if po is None:
        return 0.0
    g = _bag(go)
    p = _bag(po)
    if not g:
        return 0.0
    return sum(min(c, p.get(k, 0)) for k, c in g.items()) / sum(g.values())


def hallucination_rate(src_text, go, po):
    if po is None:
        return 1.0
    src = _norm(src_text)
    gold_values = {_norm(v) for v in _flatten(go).values()}
    pred_leaves = _flatten(po)
    denom = sum(1 for v in pred_leaves.values() if _norm(v) != "")
    if denom == 0:
        return 0.0
    unsupported = sum(
        1
        for v in pred_leaves.values()
        if _norm(v) != ""
        and not (_norm(v) in gold_values or (len(_norm(v)) >= 2 and _norm(v) in src))
    )
    return unsupported / denom


def make_reward(parse_bonus=0.1, hallucination_weight=0.75):
    def reward_func(completions, gold, text, **kwargs):
        rewards = []
        for raw, go, src in zip(completions, gold, text):
            pred, ok = parse_pred(raw)
            if not ok:
                rewards.append(-1.0)
                continue
            recall = field_recall(go, pred)
            hallucination = hallucination_rate(src, go, pred)
            rewards.append(float(recall - hallucination_weight * hallucination + parse_bonus))
        return rewards

    return reward_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="numind/NuExtract-1.5-tiny")
    parser.add_argument("--train-file", default="/workspace/grpo_augmented_train.jsonl")
    parser.add_argument("--eval-file", default="/workspace/grpo_partial_restructuring_eval.jsonl")
    parser.add_argument("--output-dir", default="/workspace/nuextract-grpo-faithfulness")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=16000)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    args = parser.parse_args()

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_file, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    config_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_steps=25,
        eval_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=25,
        beta=0.02,
        remove_unused_columns=False,
    )
    allowed = set(inspect.signature(GRPOConfig.__init__).parameters)
    config_kwargs = {k: v for k, v in config_kwargs.items() if k in allowed}
    print("GRPOConfig args:", sorted(config_kwargs))
    config = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_reward(),
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
