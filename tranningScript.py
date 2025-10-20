import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)

from peft import LoraConfig
from trl import SFTTrainer


def parse_args():
    ap = argparse.ArgumentParser(description="Chat SFT (QLoRA) for Arabic interactive assistant")
    # Data
    ap.add_argument("--chat_csv", type=str, default="full_dataset_expanded_normalized.csv",
                    help="CSV with columns: conversation_id, turn_index, role, content or content_normalized.")
    ap.add_argument("--text_col", type=str, default="content_normalized",
                    help="Column to use for text. Falls back to `content` if missing.")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (0-1).")
    ap.add_argument("--seed", type=int, default=42)

    # Model
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="Arabic-capable chat model (e.g., Qwen/Qwen2.5-7B-Instruct, tiiuae/falcon-7b-instruct).")
    ap.add_argument("--max_seq_len", type=int, default=2048)

    # LoRA / QLoRA
    ap.add_argument("--use_4bit", action="store_true", default=True,
                    help="Load base model in 4-bit (QLoRA). Pass --no_4bit to disable.")
    ap.add_argument("--no_4bit", action="store_false", dest="use_4bit")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="all-linear",
                    help="`all-linear` for auto, or comma-separated list like q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")

    # Training
    ap.add_argument("--output_dir", type=str, default="out_chat_sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=64, help="Effective batch size = batch_size * grad_accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_strategy", type=str, default="epoch",
                    choices=["no","steps","epoch"])
    ap.add_argument("--eval_strategy", type=str, default="no",
                    choices=["no","steps","epoch"])
    ap.add_argument("--bf16", action="store_true", default=torch.cuda.is_available(),
                    help="Use bfloat16 if available")
    ap.add_argument("--fp16", action="store_true", default=False,
                    help="Use fp16 (set if you don’t have bf16)")

    # Packing (concatenate multiple convos to fill sequence)
    ap.add_argument("--packing", action="store_true", default=True)
    ap.add_argument("--no_packing", action="store_false", dest="packing")

    return ap.parse_args()


def load_chat_df(csv_path: str, text_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # choose text column
    if text_col not in df.columns:
        if "content" in df.columns:
            text_col = "content"
        else:
            raise ValueError(f"Neither `{text_col}` nor `content` found in CSV. Available: {list(df.columns)}")

    needed = ["conversation_id", "turn_index", "role", text_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[needed].rename(columns={text_col: "content"})
    # Keep only valid roles & non-empty text
    df = df[df["role"].isin(["system", "user", "assistant"])].copy()
    df["content"] = df["content"].astype(str).apply(lambda s: s.strip())
    df = df[df["content"] != ""]
    # Sort
    df = df.sort_values(["conversation_id", "turn_index"]).reset_index(drop=True)
    return df


def group_to_messages(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group rows into conversations → {'messages': [ ... ]} dicts."""
    samples = []
    for cid, g in df.groupby("conversation_id", sort=False):
        msgs = [{"role": r, "content": c} for r, c in zip(g["role"], g["content"])]
        if not msgs:
            continue
        # Ensure conversation starts with system; if not, you can prepend a default system prompt
        if msgs[0]["role"] != "system":
            msgs = [{"role": "system",
                     "content": "أنت مساعد عربي للمؤسسة. أجب بدقة وفق السياسات، بلغة رسمية واضحة، وامتنع عن الاختلاق."}] + msgs
        samples.append({"id": str(cid), "messages": msgs})
    return samples


def split_train_val(samples: List[Dict[str, Any]], val_ratio=0.1, seed=42):
    import random
    rnd = random.Random(seed)
    n = len(samples)
    n_val = max(1, int(n * val_ratio))
    idx = list(range(n)); rnd.shuffle(idx)
    val_idx = set(idx[:n_val])
    train = [samples[i] for i in range(n) if i not in val_idx]
    val = [samples[i] for i in range(n) if i in val_idx]
    return train, val


def render_with_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Use the model's native chat template to render a single conversation to plain text.
    We do NOT add a generation prompt here; SFTTrainer learns to produce assistant turns.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


def make_hf_dataset(tokenizer, samples: List[Dict[str, Any]]):
    # Render each conversation to a single string using chat template
    texts = []
    for ex in samples:
        txt = render_with_chat_template(tokenizer, ex["messages"])
        texts.append(txt)

    ds = Dataset.from_dict({"text": texts})
    return ds


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CSV → conversations
    df = load_chat_df(args.chat_csv, args.text_col)
    samples = group_to_messages(df)
    train_samples, val_samples = split_train_val(samples, args.val_ratio, args.seed)

    print(f"Conversations: total={len(samples)} | train={len(train_samples)} | val={len(val_samples)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        # For many instruct LLMs, eos_token is used as pad
        tokenizer.pad_token = tokenizer.eos_token

    # Render datasets using the model's chat template
    train_ds = make_hf_dataset(tokenizer, train_samples)
    val_ds = make_hf_dataset(tokenizer, val_samples) if len(val_samples) > 0 else None
    dsdict = DatasetDict({"train": train_ds, "validation": val_ds} if val_ds else {"train": train_ds})

    # Load base model in 4bit if requested
    load_kwargs = {}
    if args.use_4bit:
        load_kwargs.update(dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16
        ))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map="auto",
        **load_kwargs
    )

    # Target modules for LoRA
    target_modules = None
    if args.target_modules != "all-linear":
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.eval_strategy if "validation" in dsdict else "no",
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        save_safetensors=True,
    )

    # NOTE: We train on the fully-rendered chat text. For a first pass, we do not mask user tokens.
    # This keeps things simple and works well for institutional SFT. Later, you can add a completion-only collator.

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=dsdict["train"],
        eval_dataset=dsdict.get("validation"),
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=args.packing,   # Concatenate multiple examples to fill sequence
        args=training_args,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save meta about splits & args
    meta = {
        "train_conversations": len(train_samples),
        "val_conversations": len(val_samples),
        "model_name": args.model_name,
        "use_4bit": args.use_4bit,
        "lora": {
            "r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
            "target_modules": args.target_modules
        },
        "max_seq_len": args.max_seq_len,
        "packing": args.packing,
    }
    with open(os.path.join(args.output_dir, "sft_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✅ Training complete. Adapter saved to:", args.output_dir)
    print("Tip: merge LoRA into the base model for export, or load with PEFT for inference.")


if __name__ == "__main__":
    main()


#  python "tranningScript.py" --chat_csv full_dataset_expanded_normalized.csv --model_name Qwen/Qwen2.5-7B-Instruct --output_dir out_chat_sft_qwen7b --epochs 2 --batch_size 1 --grad_accum 64 --lr 2e-4 --packing