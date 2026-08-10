# Task 2 - Sarcasm Headline Analysis

## Run

1. Download the [News Headlines Dataset](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) and put `Sarcasm_Headlines_Dataset_v2.json` in `data/`.
2. From this folder, run `python sarcasm_analysis.py`.

The script writes three visualizations and a short text summary to `outputs/`.

## Short answer

A simple keyword rule could catch a few sarcastic headlines, especially when a headline uses exaggerated or absurd-sounding language, but it would not work reliably overall. Many common words appear in both sources, and the same word can be literal or sarcastic depending on context. Headline length and punctuation show broad overlaps between the classes as well. A better system would need to consider combinations of words and context rather than a fixed list of keywords.
