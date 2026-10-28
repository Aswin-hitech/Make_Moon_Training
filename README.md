# Make Moon Training

This repository demonstrates a machine learning workflow for binary classification on the nonlinear `make_moons` dataset using a small PyTorch neural network. The goal of the project is to learn a curved decision boundary that separates the two crescent-shaped classes and to evaluate the model using both standard classification metrics and additional diagnostic metrics such as correlation and probability-based performance indicators.

## Approach and methodology

### 1. Dataset generation
The project uses `sklearn.datasets.make_moons`, which creates a 2D synthetic dataset with two interleaving half-moon shaped classes. This is a classic benchmark for testing whether a model can capture non-linear decision boundaries.

Key dataset settings:
- Number of samples: 1000
- Noise level: 0.2
- Random state: 42
- Split: 80% training, 20% testing

### 2. Preprocessing
Before model training, the input features are standardized using `StandardScaler`.
This is important because neural networks train more stably when feature scales are comparable, especially when using optimization methods like SGD.

### 3. Model architecture
A simple fully connected neural network is used:
- Input layer: 2 features
- Hidden layer: 16 neurons
- Activation: ReLU
- Output layer: 2 classes
- Loss function: CrossEntropyLoss
- Optimizer: SGD

This design is compact, efficient, and sufficient to model the non-linear separation between the two moons.

### 4. Training process
The model is trained for 200 epochs while tracking:
- training loss
- test loss
- training accuracy
- test accuracy

This allows the project to observe whether the model is learning effectively and whether it is overfitting or underfitting.

### 5. Evaluation strategy
After training, the model is evaluated on the test set using:
- Accuracy
- Precision
- Recall
- F1-score
- Balanced Accuracy
- ROC-AUC
- Log Loss
- Matthews Correlation Coefficient (MCC)
- Pearson correlation coefficient (r)
- Spearman rank correlation
- Confusion matrix

These metrics help assess not only whether the model is correct, but also how confident and balanced its predictions are.

## Techniques used

- Synthetic data generation using `make_moons`
- Feature scaling with standardization
- Neural network training in PyTorch
- Loss tracking across epochs
- Confusion matrix analysis
- Probability-based classification metrics
- Correlation analysis between prediction confidence and label alignment

## Main outputs

The repository generates the following outputs:

1. `MAKE_MOONS_PROJECT.ipynb`  
   Original notebook containing the full workflow, training logic, plots, and metric calculations.

2. `training_metrics.py`  
   Reusable Python script that reproduces the training and evaluation workflow in a cleaner, script-based form.

3. `metrics_summary.txt`  
   Text summary of model performance metrics.

4. `metrics_visualization.png`  
   Visualization showing loss curves, accuracy curves, and confusion matrix.

## Observed metrics

The notebook output produced the following classification results on the test set:

| Metric | Value | Interpretation |
| --- | ---: | --- |
| Accuracy | 0.8150 | 81.5% of test samples were correctly classified. |
| Precision | 0.8000 | 80.0% of predicted positive cases were correct. |
| Recall | 0.8400 | 84.0% of actual positive cases were identified. |
| F1-Score | 0.8195 | Balanced precision/recall performance. |

Additional metrics computed through the evaluation script include:
- Balanced Accuracy
- ROC-AUC
- Log Loss
- MCC
- Pearson r
- Spearman rho

These values provide a more complete assessment of the model's predictive behavior beyond simple accuracy.

## Metric interpretation

### Accuracy
Accuracy tells how often the model is correct overall. It is useful as a quick summary, though it can be less informative when the class distribution is not perfectly balanced.

### Precision
Precision indicates how reliable positive predictions are. A high value means the model rarely predicts the positive class unless it is confident.

### Recall
Recall captures how many true positive samples the model successfully detects. This is especially important when missing a positive case is costly.

### F1-score
The F1-score combines precision and recall into a single metric and is useful when a good balance between false positives and false negatives is required.

### Balanced accuracy
This metric averages recall across both classes and is helpful when class distribution is uneven or when one class is harder to predict.

### ROC-AUC
Area under the ROC curve summarizes the model's ability to separate classes across different decision thresholds.

### Log loss
Log loss evaluates the quality of predicted probabilities. Lower values indicate stronger probabilistic confidence calibration.

### MCC
The Matthews Correlation Coefficient is a robust classification metric that captures the quality of all four confusion-matrix entries and is often more informative than raw accuracy in imbalanced scenarios.

### Pearson r and Spearman rho
These correlation metrics assess how strongly the model's confidence scores align with the true labels. They are useful for understanding whether the model is not only making correct predictions, but also ranking examples in a consistent and meaningful way.

## Observations

- The model learns the non-linear structure of the dataset effectively.
- The two moon-shaped classes are separable with a simple neural network, and the model achieves strong performance on the test set.
- The results show a good balance between classification correctness and probabilistic confidence.
- While accuracy is strong, the broader metric set provides better confidence that the model is robust and not simply overfitting to one class.
- The dataset is a classic example of a task where linear models fail but shallow neural networks succeed.

## How to run

```bash
python training_metrics.py
```

This command will generate:
- `metrics_summary.txt`
- `metrics_visualization.png`

## Summary

This project introduces a complete machine learning pipeline: generating a non-linear dataset, standardizing features, training a PyTorch classifier, visualizing learning behavior, and evaluating performance with a broad set of metrics. It is a compact but effective demonstration of how neural networks handle non-linear classification problems and how model quality should be judged beyond accuracy alone.
