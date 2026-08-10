# Task 1 - Spotify Track Popularity

## Run

1. Download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and unzip `dataset.csv` into `data/`.
2. From this folder, run `python spotify_analysis.py`.

The script writes three PNG charts plus `summary.txt` to `outputs/`.

## Short answer

The strongest simple relationships with popularity are not large: in this dataset, energy and danceability have only weak linear correlations with popularity. Genre has a more noticeable association, because the average popularity differs substantially between genres, although averages can be affected by a few currently popular tracks. Tempo has almost no visible linear pattern with popularity. This makes sense because popularity depends on exposure, artist recognition, release timing, playlists, and social trends, not just audio features.
