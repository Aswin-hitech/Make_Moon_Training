# Make Moon Training

This repository contains a small PyTorch neural-network experiment on the classic `make_moons` synthetic dataset. The goal is to train a binary classifier that separates the two crescent-shaped classes and evaluate the model with standard machine-learning metrics.

## Project overview

- Dataset: `sklearn.datasets.make_moons`
- Samples: 1000
- Noise: 0.2
- Train/test split: 80/20
- Model: a 2-layer artificial neural network (input -> hidden -> output)
- Activation: ReLU
- Output layer: softmax with cross-entropy loss
- Optimizer: SGD

## Why this dataset is useful

The `make_moons` dataset is a nonlinear classification problem. A linear classifier cannot capture the curved decision boundary well, so this project is a good example of how a small neural network learns non-linear separation.

## Model architecture

The network uses:

- Input layer: 2 features
- Hidden layer: 16 neurons
- ReLU activation
- Output layer: 2 classes
- Softmax output probabilities

This setup is simple but effective for modeling the crescent boundaries of the moon-shaped classes.

## Observed training results

The existing notebook produced the following evaluation metrics on the test set:

| Metric | Value | Interpretation |
| --- | ---: | --- |
| Accuracy | 0.8150 | The model correctly classified 81.5% of the test samples. |
| Precision | 0.8000 | Of the samples predicted as the positive class, 80.0% were correct. |
| Recall | 0.8400 | The model captured 84.0% of the actual positive class samples. |
| F1-Score | 0.8195 | Balanced performance between precision and recall. |

## Additional metrics included in the evaluation code

The repository now includes a script that computes a broader set of evaluation metrics so the model can be assessed more completely:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Balanced accuracy
- ROC-AUC
- Log loss
- MCC (Matthews correlation coefficient)
- Pearson correlation coefficient (r)
- Spearman rank correlation

## What the metrics mean

### Accuracy
Accuracy tells how often the model is correct overall. It is useful for a general summary, but it can be misleading when class balance is uneven.

### Precision
Precision measures how reliable positive predictions are. A high precision means the model does not label many samples as positive unless it is confident.

### Recall
Recall measures how many true positive cases the model successfully catches. A high recall is important when false negatives are costly.

### F1-score
The F1-score balances precision and recall. It is especially useful when the model is evaluated on a single positive class and class imbalance may affect accuracy.

### ROC-AUC
AUC summarizes the model's ability to rank a positive sample above a negative sample across all thresholds.

### Log loss
Log loss measures confidence calibration. Lower values indicate better probabilistic predictions.

### MCC
The Matthews correlation coefficient is a balanced classification metric that stays informative even when class distribution is not equal.

### Pearson r / correlation
Pearson correlation measures the linear relationship between the model's positive-class probability scores and the true labels. This is useful when assessing how strongly the model's confidence aligns with actual class membership.

### Spearman correlation
Spearman rank correlation checks whether the ordering of scores follows the order of the labels, even when the relationship is not strictly linear.

## Observations

- The model learns the nonlinear structure of the `make_moons` dataset well enough to achieve strong classification performance.
- The test metrics indicate a good balance between correctly identifying positive examples and avoiding false alarms.
- The structure of the dataset is challenging because the classes are intertwined; therefore, a modest but meaningful error rate is expected even for a well-performing model.
- Accuracy alone is not enough for a full model assessment; the additional metrics help show confidence, balance, and ranking quality.

## Run the evaluation

Use the repository script to reproduce the analysis and export the updated metrics summary and plot:

```bash
python training_metrics.py
```

This will generate:

- `metrics_summary.txt`
- `metrics_visualization.png`

## Repository files

- `MAKE_MOONS_PROJECT.ipynb` — original notebook implementation
- `training_metrics.py` — reusable script for training and full metric evaluation
- `metrics_summary.txt` — current empirical metrics summary
- `metrics_visualization.png` — training/loss and metric visualization

## Summary

This project demonstrates a clean workflow for training a binary classifier on a non-linear dataset: generate data, standardize features, train a small neural network, evaluate the model with a rich set of metrics, and communicate the results clearly.
