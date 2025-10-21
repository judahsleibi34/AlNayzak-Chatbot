import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict
import torch
import numpy as np
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


class OverfittingDetectionCallback(TrainerCallback):
    """
    Detects overfitting by monitoring train/val loss divergence.
    Stops training early if overfitting is detected.
    """
    def __init__(self, patience: int = 3, min_delta: float = 0.01, divergence_threshold: float = 0.15):
        """
        Args:
            patience: Number of evaluations to wait before stopping
            min_delta: Minimum change to consider as improvement
            divergence_threshold: Max acceptable gap between train and val loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.divergence_threshold = divergence_threshold
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.should_stop = False
        
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return control
            
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return control
            
        # Get recent training loss
        train_loss = state.log_history[-2].get("loss") if len(state.log_history) >= 2 else None
        
        if train_loss is not None:
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Check for divergence (overfitting signal)
            loss_gap = val_loss - train_loss
            if loss_gap > self.divergence_threshold:
                print(f"\n⚠️  Overfitting detected: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, gap={loss_gap:.4f}")
                self.patience_counter += 1
            else:
                # Check if validation loss improved
                if val_loss < (self.best_val_loss - self.min_delta):
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    print(f"✓ Val loss improved to {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"→ Val loss: {val_loss:.4f} (no improvement, patience: {self.patience_counter}/{self.patience})")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n🛑 Early stopping triggered after {self.patience} evaluations without improvement")
                control.should_training_stop = True
                self.should_stop = True
                
        return control


def parse_args():
    ap = argparse.ArgumentParser(description="Robust Chat SFT (QLoRA) with overfitting detection")
    # Data
    ap.add_argument("--chat_csv", type=str, default="full_dataset_expanded_normalized.csv",
                    help="CSV with columns: conversation_id, turn_index, role, content or content_normalized.")
    ap.add_argument("--text_col", type=str, default="content_normalized",
                    help="Column to use for text. Falls back to `content` if missing.")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (0-1).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_convo_length", type=int, default=2,
                    help="Minimum number of turns per conversation")

    # Model
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="Arabic-capable chat model")
    ap.add_argument("--max_seq_len", type=int, default=2048)

    # LoRA / QLoRA
    ap.add_argument("--use_4bit", action="store_true", default=True,
                    help="Load base model in 4-bit (QLoRA)")
    ap.add_argument("--no_4bit", action="store_false", dest="use_4bit")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="all-linear")

    # Training
    ap.add_argument("--output_dir", type=str, default="out_chat_sft")
    ap.add_argument("--epochs", type=int, default=10, help="Max epochs (early stopping may end sooner)")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--eval_steps", type=int, default=100,
                    help="Evaluate every N steps (for early stopping)")
    ap.add_argument("--save_strategy", type=str, default="steps")
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=3,
                    help="Keep only best N checkpoints")
    ap.add_argument("--bf16", action="store_true", default=torch.cuda.is_available())
    ap.add_argument("--fp16", action="store_true", default=False)

    # Early stopping / Overfitting detection
    ap.add_argument("--early_stopping_patience", type=int, default=3,
                    help="Stop after N evals without improvement")
    ap.add_argument("--divergence_threshold", type=float, default=0.15,
                    help="Max acceptable train/val loss gap")
    ap.add_argument("--min_delta", type=float, default=0.01,
                    help="Minimum improvement to reset patience")

    # Packing
    ap.add_argument("--packing", action="store_true", default=True)
    ap.add_argument("--no_packing", action="store_false", dest="packing")

    ap.add_argument("--cache_dir", type=str, default="D:/hf_cache", help="Directory to cache models and tokenizers")
    
    return ap.parse_args()


def validate_dataframe(df: pd.DataFrame, text_col: str, min_convo_length: int) -> pd.DataFrame:
    """Validate and clean dataframe with detailed reporting."""
    initial_rows = len(df)
    initial_convos = df["conversation_id"].nunique()
    
    print(f"\n📊 Dataset validation:")
    print(f"  Initial: {initial_rows} rows, {initial_convos} conversations")
    
    # Check for missing values
    missing = df[["conversation_id", "turn_index", "role", "content"]].isnull().sum()
    if missing.any():
        print(f"  ⚠️  Missing values detected:\n{missing[missing > 0]}")
        df = df.dropna(subset=["conversation_id", "turn_index", "role", "content"])
        print(f"  Dropped rows with missing values: {initial_rows - len(df)}")
    
    # Remove duplicate turns
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["conversation_id", "turn_index"], keep="first")
    if before_dedup != len(df):
        print(f"  Removed {before_dedup - len(df)} duplicate turns")
    
    # Filter short conversations
    convo_lengths = df.groupby("conversation_id").size()
    valid_convos = convo_lengths[convo_lengths >= min_convo_length].index
    df = df[df["conversation_id"].isin(valid_convos)]
    
    removed_convos = initial_convos - len(valid_convos)
    if removed_convos > 0:
        print(f"  Removed {removed_convos} conversations with < {min_convo_length} turns")
    
    # Check role distribution
    role_counts = df["role"].value_counts()
    print(f"  Role distribution:\n{role_counts.to_string()}")
    
    # Check content length statistics
    df["content_length"] = df["content"].str.len()
    print(f"  Content length stats:")
    print(f"    Mean: {df['content_length'].mean():.0f} chars")
    print(f"    Median: {df['content_length'].median():.0f} chars")
    print(f"    Min: {df['content_length'].min()}, Max: {df['content_length'].max()}")
    
    final_rows = len(df)
    final_convos = df["conversation_id"].nunique()
    print(f"  Final: {final_rows} rows, {final_convos} conversations")
    
    if final_convos == 0:
        raise ValueError("No valid conversations remaining after filtering!")
    
    return df.drop(columns=["content_length"])


def load_chat_df(csv_path: str, text_col: str, min_convo_length: int) -> pd.DataFrame:
    """Load and validate CSV with robust error handling."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"❌ Error reading CSV with utf-8-sig encoding: {e}")
        print("Trying alternate encodings...")
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                print(f"✓ Successfully loaded with {enc} encoding")
                break
            except:
                continue
        else:
            raise ValueError(f"Could not read CSV with any supported encoding")
    
    # Choose text column
    if text_col not in df.columns:
        if "content" in df.columns:
            print(f"⚠️  Column '{text_col}' not found, using 'content' instead")
            text_col = "content"
        else:
            raise ValueError(f"Neither '{text_col}' nor 'content' found. Available: {list(df.columns)}")

    needed = ["conversation_id", "turn_index", "role", text_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[needed].rename(columns={text_col: "content"})
    
    # Filter valid roles
    valid_roles = ["system", "user", "assistant"]
    invalid_roles = df[~df["role"].isin(valid_roles)]["role"].unique()
    if len(invalid_roles) > 0:
        print(f"⚠️  Found invalid roles (will be removed): {invalid_roles}")
    df = df[df["role"].isin(valid_roles)].copy()
    
    # Clean content
    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"] != ""]
    
    # Sort
    df = df.sort_values(["conversation_id", "turn_index"]).reset_index(drop=True)
    
    # Validate
    df = validate_dataframe(df, text_col, min_convo_length)
    
    return df


def group_to_messages(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group rows into conversations with validation."""
    samples = []
    issues = {"no_assistant": 0, "invalid_sequence": 0}
    
    for cid, g in df.groupby("conversation_id", sort=False):
        msgs = [{"role": r, "content": c} for r, c in zip(g["role"], g["content"])]
        if not msgs:
            continue
        
        # Check if conversation has at least one assistant response
        if not any(m["role"] == "assistant" for m in msgs):
            issues["no_assistant"] += 1
            continue
        
        # Ensure conversation starts with system prompt
        if msgs[0]["role"] != "system":
            msgs = [{
                "role": "system",
                "content": "أنت مساعد عربي للمؤسسة. أجب بدقة وفق السياسات، بلغة رسمية واضحة، وامتنع عن الاختلاق."
            }] + msgs
        
        samples.append({"id": str(cid), "messages": msgs})
    
    if issues["no_assistant"] > 0:
        print(f"⚠️  Skipped {issues['no_assistant']} conversations without assistant responses")
    
    return samples


def split_train_val(samples: List[Dict[str, Any]], val_ratio=0.1, seed=42):
    """Split with stratification by conversation length if possible."""
    import random
    rnd = random.Random(seed)
    n = len(samples)
    
    if n == 0:
        raise ValueError("No samples to split!")
    
    n_val = max(1, int(n * val_ratio))
    n_val = min(n_val, n - 1)  # Ensure at least 1 training sample
    
    idx = list(range(n))
    rnd.shuffle(idx)
    val_idx = set(idx[:n_val])
    
    train = [samples[i] for i in range(n) if i not in val_idx]
    val = [samples[i] for i in range(n) if i in val_idx]
    
    return train, val


def render_with_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Render conversation using model's chat template with error handling."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    except Exception as e:
        print(f"⚠️  Error rendering chat template: {e}")
        # Fallback to simple concatenation
        return "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])


def make_hf_dataset(tokenizer, samples: List[Dict[str, Any]]):
    """Create HF dataset with validation."""
    texts = []
    skipped = 0
    
    for ex in samples:
        try:
            txt = render_with_chat_template(tokenizer, ex["messages"])
            if txt and len(txt.strip()) > 0:
                texts.append(txt)
            else:
                skipped += 1
        except Exception as e:
            print(f"⚠️  Error processing conversation {ex.get('id', 'unknown')}: {e}")
            skipped += 1
    
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} conversations due to rendering errors")
    
    if len(texts) == 0:
        raise ValueError("No valid conversations after rendering!")
    
    ds = Dataset.from_dict({"text": texts})
    return ds


def main():
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Robust Chat SFT with Overfitting Detection")
    print("=" * 60)
    
    # Load and validate data
    try:
        df = load_chat_df(args.chat_csv, args.text_col, args.min_convo_length)
        samples = group_to_messages(df)
        train_samples, val_samples = split_train_val(samples, args.val_ratio, args.seed)
    except Exception as e:
        print(f"\n❌ Data loading failed: {e}")
        raise

    print(f"\n📈 Dataset split:")
    print(f"  Total conversations: {len(samples)}")
    print(f"  Training: {len(train_samples)} ({len(train_samples)/len(samples)*100:.1f}%)")
    print(f"  Validation: {len(val_samples)} ({len(val_samples)/len(samples)*100:.1f}%)")

    # Load tokenizer
    print(f"\n🔤 Loading tokenizer: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            use_fast=True,
            cache_dir=args.cache_dir if args.cache_dir else None
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("  Set pad_token = eos_token")
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        raise

    # Create datasets
    print("\n📝 Rendering conversations with chat template...")
    try:
        train_ds = make_hf_dataset(tokenizer, train_samples)
        val_ds = make_hf_dataset(tokenizer, val_samples) if val_samples else None
        dsdict = DatasetDict({"train": train_ds, "validation": val_ds} if val_ds else {"train": train_ds})
        print(f"  Train examples: {len(train_ds)}")
        if val_ds:
            print(f"  Validation examples: {len(val_ds)}")
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        raise

    # Load model
    print(f"\n🤖 Loading model: {args.model_name}")
    load_kwargs = {"cache_dir": args.cache_dir if args.cache_dir else None}
    if args.use_4bit:
        print("  Using 4-bit quantization (QLoRA)")
        load_kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.bfloat16 if args.bf16 else torch.float16
        })

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else None,
            device_map="auto",
            **load_kwargs
        )
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise

    # LoRA config
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
    
    print(f"\n⚙️  LoRA Configuration:")
    print(f"  r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  Target modules: {args.target_modules}")

    # SFTConfig (replaces TrainingArguments + adds SFT-specific params)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if "validation" in dsdict else "no",
        load_best_model_at_end=True if "validation" in dsdict else False,
        metric_for_best_model="eval_loss" if "validation" in dsdict else None,
        greater_is_better=False,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        save_safetensors=True,
        seed=args.seed,
        # SFT-specific parameters (moved from SFTTrainer)
        max_length=args.max_seq_len,  # Changed from max_seq_length to max_length
        packing=args.packing,
        dataset_text_field="text",  # Our dataset has a 'text' field
    )

    # Initialize trainer with overfitting detection
    callbacks = []
    if "validation" in dsdict:
        overfitting_callback = OverfittingDetectionCallback(
            patience=args.early_stopping_patience,
            min_delta=args.min_delta,
            divergence_threshold=args.divergence_threshold
        )
        callbacks.append(overfitting_callback)
        print(f"\n🎯 Early stopping enabled:")
        print(f"  Patience: {args.early_stopping_patience} evaluations")
        print(f"  Divergence threshold: {args.divergence_threshold}")
        print(f"  Min delta: {args.min_delta}")
  
    trainer = SFTTrainer(
        model=model,
        args=sft_config,  # Changed from training_args to args with SFTConfig
        train_dataset=dsdict["train"],
        eval_dataset=dsdict.get("validation"),
        processing_class=tokenizer,  # SFTTrainer uses processing_class
        peft_config=peft_config,
        callbacks=callbacks,
    )

    # Train
    print("\n" + "=" * 60)
    print("🏋️  Starting training...")
    print("=" * 60 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

    # Save final model
    print("\n💾 Saving model and tokenizer...")
    try:
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"❌ Saving failed: {e}")
        raise

    # Save metadata
    meta = {
        "data": {
            "train_conversations": len(train_samples),
            "val_conversations": len(val_samples),
            "source_csv": args.chat_csv,
            "text_column": args.text_col,
        },
        "model": {
            "base_model": args.model_name,
            "use_4bit": args.use_4bit,
            "max_seq_len": args.max_seq_len,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.target_modules,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "learning_rate": args.lr,
            "packing": args.packing,
        },
        "early_stopping": {
            "enabled": len(callbacks) > 0,
            "patience": args.early_stopping_patience if len(callbacks) > 0 else None,
            "divergence_threshold": args.divergence_threshold if len(callbacks) > 0 else None,
        }
    }
    
    # Add training history if available
    if len(callbacks) > 0 and hasattr(callbacks[0], 'train_losses'):
        meta["training_history"] = {
            "train_losses": [float(x) for x in callbacks[0].train_losses],
            "val_losses": [float(x) for x in callbacks[0].val_losses],
            "stopped_early": callbacks[0].should_stop,
        }
    
    meta_path = output_path / "sft_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Metadata saved to: {meta_path}")
    print("\n💡 Next steps:")
    print("  1. Merge LoRA with base model: Use PEFT merge_and_unload()")
    print("  2. Run inference: Load adapter with PEFT or merged model")
    print("  3. Evaluate: Test on held-out data or with human evaluation")


if __name__ == "__main__":
    main()

# python tranningScriptTwo.py --chat_csv full_dataset_expanded_normalized.csv --model_name Qwen/Qwen2.5-7B-Instruct --output_dir out_chat_sft_qwen7b --epochs 30 --batch_size 1 --grad_accum 64 --lr 2e-4 --packing