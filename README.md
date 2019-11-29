# Project 2 - To Be or Not to Be

Using 'bag of words' feature engineering and decision trees to predict players in Shakespeare plays.

## Dataset

https://www.kaggle.com/kingburrito666/shakespeare-plays

## Feature Engineering

* Rows without act-scene-line are ignored, since they are not actual spoken lines
* 'Bag of Words' vector encoding on text data
* One-hot encoding for play
* Labeling for act-scene-line data

## Training

* 70 / 30 split of training to test data
* Decision Tree scores 72% accuracy