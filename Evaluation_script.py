import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def parse_args():
    ap = argparse.ArgumentParser(description="Comprehensive evaluation of trained chat model")
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", type=str, required=True,
                    help="Path to trained LoRA adapter")
    ap.add_argument("--test_csv", type=str, required=True,
                    help="CSV file with test conversations")
    ap.add_argument("--text_col", type=str, default="content_normalized")
    ap.add_argument("--output_dir", type=str, default="evaluation_results")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Limit number of test samples (for quick testing)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_test_data(csv_path: str, text_col: str) -> pd.DataFrame:
    """Load test conversations from CSV."""
    print(f"📂 Loading test data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except:
        df = pd.read_csv(csv_path, encoding="utf-8")
    
    if text_col not in df.columns and "content" in df.columns:
        text_col = "content"
    
    required = ["conversation_id", "turn_index", "role", text_col]
    df = df[required].rename(columns={text_col: "content"})
    
    df = df[df["role"].isin(["system", "user", "assistant"])].copy()
    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"] != ""]
    df = df.sort_values(["conversation_id", "turn_index"]).reset_index(drop=True)
    
    print(f"  Loaded {len(df)} turns from {df['conversation_id'].nunique()} conversations")
    return df


def create_test_examples(df: pd.DataFrame) -> List[Dict]:
    """Create test examples: given context, predict next assistant response."""
    examples = []
    
    for conv_id, group in df.groupby("conversation_id"):
        msgs = [{"role": r, "content": c} for r, c in zip(group["role"], group["content"])]
        
        # Find all assistant responses to evaluate
        for i, msg in enumerate(msgs):
            if msg["role"] == "assistant":
                # Context is everything before this assistant response
                context = msgs[:i]
                if len(context) == 0:
                    continue
                
                # Ensure context starts with system
                if context[0]["role"] != "system":
                    context = [{
                        "role": "system",
                        "content": "أنت مساعد عربي للمؤسسة. أجب بدقة وفق السياسات، بلغة رسمية واضحة، وامتنع عن الاختلاق."
                    }] + context
                
                examples.append({
                    "conversation_id": conv_id,
                    "turn_index": i,
                    "context": context,
                    "reference": msg["content"]
                })
    
    return examples


def load_model_and_tokenizer(base_model: str, adapter_path: str, device: str):
    """Load base model with LoRA adapter."""
    print(f"\n🤖 Loading model...")
    print(f"  Base: {base_model}")
    print(f"  Adapter: {adapter_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if device == "cuda" else "cpu",  # <-- Fixed
        low_cpu_mem_usage=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # Merge for faster inference
    model.eval()
    
    print(f"  ✓ Model loaded on {device}")
    return model, tokenizer


def generate_response(model, tokenizer, context: List[Dict], max_length: int, device: str) -> str:
    """Generate model response given context."""
    # Format context with chat template
    prompt = tokenizer.apply_chat_template(
        context,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # Greedy decoding for reproducibility
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def calculate_token_accuracy(model, tokenizer, examples: List[Dict], device: str, max_samples: int = None) -> Dict:
    """Calculate next-token prediction accuracy."""
    print("\n📊 Calculating token-level accuracy...")
    
    if max_samples:
        examples = examples[:max_samples]
    
    all_correct = 0
    all_total = 0
    
    for ex in tqdm(examples, desc="Token accuracy"):
        # Create full conversation including reference
        full_conv = ex["context"] + [{"role": "assistant", "content": ex["reference"]}]
        
        # Tokenize
        text = tokenizer.apply_chat_template(full_conv, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
        
        # Get predictions (shift by 1 for next-token prediction)
        predictions = torch.argmax(logits, dim=-1)
        labels = inputs["input_ids"]
        
        # Compare predictions with labels (shift alignment)
        pred_tokens = predictions[:, :-1]
        label_tokens = labels[:, 1:]
        
        correct = (pred_tokens == label_tokens).sum().item()
        total = label_tokens.numel()
        
        all_correct += correct
        all_total += total
    
    accuracy = all_correct / all_total if all_total > 0 else 0
    
    print(f"  Token Accuracy: {accuracy:.4f} ({all_correct}/{all_total})")
    return {
        "token_accuracy": accuracy,
        "correct_tokens": all_correct,
        "total_tokens": all_total
    }


def calculate_response_metrics(examples: List[Dict], predictions: List[str]) -> Dict:
    """Calculate response-level metrics."""
    print("\n📊 Calculating response-level metrics...")
    
    # For classification metrics, we need binary labels
    # We'll use a simple heuristic: is the response non-empty and relevant?
    
    metrics = {
        "total_responses": len(predictions),
        "empty_responses": sum(1 for p in predictions if len(p.strip()) == 0),
        "avg_response_length": np.mean([len(p) for p in predictions]),
        "median_response_length": np.median([len(p) for p in predictions]),
    }
    
    # Calculate exact match accuracy (rare in generation)
    exact_matches = sum(1 for ex, pred in zip(examples, predictions) 
                       if ex["reference"].strip() == pred.strip())
    metrics["exact_match_accuracy"] = exact_matches / len(examples) if examples else 0
    
    # Calculate BLEU-like overlap (simplified)
    word_overlaps = []
    for ex, pred in zip(examples, predictions):
        ref_words = set(ex["reference"].lower().split())
        pred_words = set(pred.lower().split())
        if len(ref_words) > 0:
            overlap = len(ref_words & pred_words) / len(ref_words)
            word_overlaps.append(overlap)
    
    metrics["avg_word_overlap"] = np.mean(word_overlaps) if word_overlaps else 0
    
    print(f"  Total responses: {metrics['total_responses']}")
    print(f"  Empty responses: {metrics['empty_responses']}")
    print(f"  Exact matches: {exact_matches}")
    print(f"  Avg word overlap: {metrics['avg_word_overlap']:.4f}")
    print(f"  Avg response length: {metrics['avg_response_length']:.1f} chars")
    
    return metrics


def generate_predictions(model, tokenizer, examples: List[Dict], device: str, 
                        max_length: int, max_samples: int = None) -> List[str]:
    """Generate predictions for all test examples."""
    print("\n🔮 Generating predictions...")
    
    if max_samples:
        examples = examples[:max_samples]
    
    predictions = []
    
    for ex in tqdm(examples, desc="Generating"):
        try:
            pred = generate_response(model, tokenizer, ex["context"], max_length, device)
            predictions.append(pred)
        except Exception as e:
            print(f"⚠️  Error generating response: {e}")
            predictions.append("")
    
    return predictions


def save_predictions(examples: List[Dict], predictions: List[str], output_dir: Path):
    """Save predictions with references for manual inspection."""
    results = []
    
    for ex, pred in zip(examples, predictions):
        results.append({
            "conversation_id": ex["conversation_id"],
            "turn_index": ex["turn_index"],
            "reference": ex["reference"],
            "prediction": pred,
            "word_overlap": len(set(ex["reference"].lower().split()) & 
                              set(pred.lower().split())) / max(len(ex["reference"].split()), 1)
        })
    
    df = pd.DataFrame(results)
    output_path = output_dir / "predictions.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Predictions saved to: {output_path}")
    
    return df


def plot_metrics(metrics: Dict, output_dir: Path):
    """Create visualization of evaluation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Response length distribution
    if "response_lengths" in metrics:
        axes[0, 0].hist(metrics["response_lengths"], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel("Response Length (characters)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Distribution of Response Lengths")
        axes[0, 0].grid(True, alpha=0.3)
    
    # Word overlap distribution
    if "word_overlaps" in metrics:
        axes[0, 1].hist(metrics["word_overlaps"], bins=50, alpha=0.7, color='green')
        axes[0, 1].set_xlabel("Word Overlap Score")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of Word Overlap with References")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Metrics bar chart
    metric_names = ["Token\nAccuracy", "Exact\nMatch", "Avg Word\nOverlap"]
    metric_values = [
        metrics.get("token_accuracy", 0),
        metrics.get("exact_match_accuracy", 0),
        metrics.get("avg_word_overlap", 0)
    ]
    
    axes[1, 0].bar(metric_names, metric_values, alpha=0.7, color=['blue', 'orange', 'green'])
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Summary Metrics")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, v in enumerate(metric_values):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Empty responses pie chart
    empty_count = metrics.get("empty_responses", 0)
    non_empty_count = metrics.get("total_responses", 0) - empty_count
    
    if empty_count + non_empty_count > 0:
        axes[1, 1].pie(
            [non_empty_count, empty_count],
            labels=['Valid Responses', 'Empty Responses'],
            autopct='%1.1f%%',
            colors=['lightgreen', 'lightcoral'],
            startangle=90
        )
        axes[1, 1].set_title("Response Validity")
    
    plt.tight_layout()
    plot_path = output_dir / "evaluation_metrics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Metrics visualization saved to: {plot_path}")


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🧪 Comprehensive Model Evaluation")
    print("=" * 60)
    
    # Load test data
    test_df = load_test_data(args.test_csv, args.text_col)
    test_examples = create_test_examples(test_df)
    
    if args.max_samples:
        test_examples = test_examples[:args.max_samples]
        print(f"⚠️  Limited to {args.max_samples} samples for evaluation")
    
    print(f"\n📝 Test examples: {len(test_examples)}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.base_model,
        args.adapter_path,
        args.device
    )
    
    # Generate predictions
    predictions = generate_predictions(
        model, tokenizer, test_examples,
        args.device, args.max_length, args.max_samples
    )
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("📊 Computing Evaluation Metrics")
    print("=" * 60)
    
    # Token-level accuracy
    token_metrics = calculate_token_accuracy(
        model, tokenizer, test_examples,
        args.device, args.max_samples
    )
    
    # Response-level metrics
    response_metrics = calculate_response_metrics(test_examples, predictions)
    
    # Combine all metrics
    all_metrics = {**token_metrics, **response_metrics}
    
    # Add distribution data for plotting
    all_metrics["response_lengths"] = [len(p) for p in predictions]
    all_metrics["word_overlaps"] = [
        len(set(ex["reference"].lower().split()) & set(pred.lower().split())) / 
        max(len(ex["reference"].split()), 1)
        for ex, pred in zip(test_examples, predictions)
    ]
    
    # Save predictions
    pred_df = save_predictions(test_examples, predictions, output_dir)
    
    # Plot metrics
    plot_metrics(all_metrics, output_dir)
    
    # Save metrics to JSON
    metrics_to_save = {k: v for k, v in all_metrics.items() 
                      if not isinstance(v, (list, np.ndarray))}
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("✅ Evaluation Complete - Summary")
    print("=" * 60)
    print(f"\n📊 Key Metrics:")
    print(f"  Token Accuracy:      {all_metrics['token_accuracy']:.4f}")
    print(f"  Exact Match Rate:    {all_metrics['exact_match_accuracy']:.4f}")
    print(f"  Avg Word Overlap:    {all_metrics['avg_word_overlap']:.4f}")
    print(f"  Empty Responses:     {all_metrics['empty_responses']}/{all_metrics['total_responses']}")
    print(f"  Avg Response Length: {all_metrics['avg_response_length']:.1f} chars")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"  • Predictions: predictions.csv")
    print(f"  • Metrics: evaluation_metrics.json")
    print(f"  • Visualizations: evaluation_metrics.png")
    
    print("\n💡 Next steps:")
    print("  1. Review predictions.csv for qualitative analysis")
    print("  2. Check evaluation_metrics.png for performance patterns")
    print("  3. Identify failure cases and improvement opportunities")
    print("  4. Consider human evaluation for semantic quality")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

# python tranningScript.py --chat_csv full_dataset_expanded_normalized.csv --model_name Qwen/Qwen2.5-7B-Instruct --output_dir out_chat_sft_qwen7b --epochs 30 --batch_size 1 --grad_accum 64 --lr 2e-4 --packing --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

# python tranningScript.py --chat_csv full_dataset_expanded_normalized.csv --model_name Qwen/Qwen2.5-7B-Instruct --output_dir out_chat_sft_qwen7b --epochs 30 --batch_size 1 --grad_accum 64 --lr 2e-4 --packing --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1