import pandas as pd, json

IN = "full_dataset_expanded_normalized.csv"
OUT = "chat_finetune.jsonl"

df = pd.read_csv(IN, encoding="utf-8-sig")

# Pick text column automatically
text_col = "content_normalized" if "content_normalized" in df.columns else "content"

# Keep the key chat columns
df = df[["conversation_id","turn_index","role",text_col]].rename(columns={text_col:"content"})
df = df.sort_values(["conversation_id","turn_index"])

# Group into conversation messages
with open(OUT, "w", encoding="utf-8") as f:
    for cid, g in df.groupby("conversation_id", sort=False):
        msgs = [{"role": r, "content": str(c)} for r,c in zip(g["role"], g["content"]) if str(c).strip()]
        if msgs:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

print(f"✅ Saved ready-to-train JSONL file to: {OUT}")
