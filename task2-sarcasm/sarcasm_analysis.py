"""Explores simple linguistic differences in The Onion and HuffPost headlines."""
from pathlib import Path
from collections import Counter
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "Sarcasm_Headlines_Dataset_v2.json"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
if not DATA.exists():
    raise FileNotFoundError("Put Sarcasm_Headlines_Dataset_v2.json in task2-sarcasm/data/ and run again.")

sns.set_theme(style="whitegrid")
df = pd.read_json(DATA, lines=True).dropna(subset=["headline", "is_sarcastic"])
df["class"] = df["is_sarcastic"].map({0: "Genuine (HuffPost)", 1: "Sarcastic (The Onion)"})
df["word_count"] = df["headline"].str.findall(r"\b\w+\b").str.len()
df["char_count"] = df["headline"].str.len()
df["exclamation_count"] = df["headline"].str.count("!")
df["question_count"] = df["headline"].str.count(r"\?")

# 1. Distribution of headline length.
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="word_count", hue="class", bins=30, stat="density", common_norm=False, element="step")
plt.xlim(0, df["word_count"].quantile(.99)); plt.title("Headline word-count distributions")
plt.xlabel("Words in headline")
plt.tight_layout(); plt.savefig(OUT / "01_headline_length.png", dpi=180); plt.close()

# 2. Punctuation patterns by class.
punct = df.groupby("class", as_index=False)[["exclamation_count", "question_count"]].mean().melt(id_vars="class", var_name="punctuation", value_name="average_count")
plt.figure(figsize=(8, 5))
sns.barplot(data=punct, x="punctuation", y="average_count", hue="class")
plt.title("Average punctuation marks per headline")
plt.xlabel(""); plt.ylabel("Average count")
plt.tight_layout(); plt.savefig(OUT / "02_punctuation.png", dpi=180); plt.close()

# 3. Compare the most distinctive common words after removing basic stopwords.
stop = {"the", "a", "an", "and", "of", "to", "in", "for", "on", "with", "is", "at", "from", "by", "that", "this", "as", "it", "be", "are", "new"}
def common_words(series):
    words = re.findall(r"[a-z']+", " ".join(series).lower())
    return Counter(w for w in words if w not in stop).most_common(10)
words = []
for label, group in df.groupby("class"):
    words.extend({"class": label, "word": w, "count": c} for w, c in common_words(group["headline"]))
word_df = pd.DataFrame(words)
g = sns.catplot(data=word_df, x="count", y="word", col="class", kind="bar", sharey=False, height=5, aspect=.9, color="#4C78A8")
g.set_titles("{col_name}"); g.set_axis_labels("Count", "Word")
g.figure.suptitle("Ten most common content words by class", y=1.04)
g.figure.savefig(OUT / "03_common_words.png", dpi=180, bbox_inches="tight"); plt.close(g.figure)

summary = df.groupby("class")[["word_count", "char_count", "exclamation_count", "question_count"]].mean().round(2)
with open(OUT / "summary.txt", "w", encoding="utf-8") as f:
    f.write("Mean text statistics by class:\n" + summary.to_string() + "\n\n")
    f.write("Interpretation: a keyword-only rule will catch a few patterns but cannot reliably detect sarcasm because language and punctuation overlap heavily between classes.\n")
print(f"Done. Charts and summary saved in {OUT}")
