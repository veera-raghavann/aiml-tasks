# Task 3 - Cat vs. Dog Image Classifier

Open `cat_dog_classifier.ipynb` in Jupyter or Google Colab and run all cells.

## Data setup

1. Download the [Dogs vs. Cats competition data](https://www.kaggle.com/c/dogs-vs-cats/data).
2. Unzip `train.zip` into `data/train/`, so that files such as `cat.0.jpg` and `dog.0.jpg` are directly inside that folder.
3. The notebook uses the filename prefixes to make `data/cats/` and `data/dogs/` automatically. It then trains a small CNN for 3 epochs.

The notebook prints training/validation accuracy, saves a learning-curve chart in `outputs/`, and displays five predictions. It also finds and displays the first incorrect validation prediction, when one exists, with a short possible explanation.

## Notes

This uses a small CNN rather than a larger pretrained model so it is easier to understand and run on a basic GPU or Colab session. More epochs or MobileNet transfer learning would likely improve accuracy.
