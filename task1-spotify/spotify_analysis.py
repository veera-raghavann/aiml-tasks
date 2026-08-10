"""A compact exploratory analysis for the Spotify track popularity task."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "dataset.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError("Put Kaggle's dataset.csv in task1-spotify/data/ and run again.")

sns.set_theme(style="whitegrid", palette="deep")
df = pd.read_csv(DATA)

# Basic cleaning: discard duplicate tracks, impute numeric values, and remove rows missing labels.
df = df.drop(columns=["Unnamed: 0"], errors="ignore").drop_duplicates(subset="track_id")
numeric = df.select_dtypes(include="number").columns
df[numeric] = df[numeric].fillna(df[numeric].median())
df = df.dropna(subset=["popularity", "track_genre"])

# 1. Popularity versus energy and danceability (sampling keeps the chart readable).
sample = df.sample(min(6000, len(df)), random_state=42)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.regplot(data=sample, x="energy", y="popularity", scatter_kws={"alpha": .16, "s": 12}, line_kws={"color": "crimson"}, ax=axes[0])
axes[0].set_title("Popularity vs. energy")
sns.regplot(data=sample, x="danceability", y="popularity", scatter_kws={"alpha": .16, "s": 12}, line_kws={"color": "crimson"}, ax=axes[1])
axes[1].set_title("Popularity vs. danceability")
fig.tight_layout(); fig.savefig(OUT / "01_audio_features.png", dpi=180); plt.close(fig)

# 2. Mean popularity by genre (only show the 15 highest for legibility).
genre = df.groupby("track_genre", as_index=False)["popularity"].mean().sort_values("popularity", ascending=False).head(15)
plt.figure(figsize=(10, 6))
sns.barplot(data=genre, y="track_genre", x="popularity", hue="track_genre", legend=False)
plt.title("15 genres with highest mean track popularity")
plt.xlabel("Mean popularity"); plt.ylabel("Genre")
plt.tight_layout(); plt.savefig(OUT / "02_genre_popularity.png", dpi=180); plt.close()

# 3. Tempo groups make the lack of a strong tempo effect easy to compare.
df["tempo_band"] = pd.cut(df["tempo"], bins=[0, 80, 100, 120, 140, 300], include_lowest=True)
tempo = df.groupby("tempo_band", observed=True, as_index=False)["popularity"].mean()
plt.figure(figsize=(9, 5))
sns.barplot(data=tempo, x="tempo_band", y="popularity", color="#4C78A8")
plt.xticks(rotation=20); plt.title("Mean popularity by tempo band")
plt.xlabel("Tempo (BPM)"); plt.ylabel("Mean popularity")
plt.tight_layout(); plt.savefig(OUT / "03_tempo_bands.png", dpi=180); plt.close()

corr = df[["popularity", "tempo", "energy", "danceability", "acousticness", "valence"]].corr(numeric_only=True)["popularity"].sort_values(ascending=False)
with open(OUT / "summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Rows after cleaning: {len(df):,}\n\n")
    f.write("Correlations with popularity:\n" + corr.to_string() + "\n\n")
    f.write("Interpretation: audio-feature correlations are weak; genre averages vary more, but popularity is also strongly shaped by non-audio factors.\n")
print(f"Done. Charts and summary saved in {OUT}")
