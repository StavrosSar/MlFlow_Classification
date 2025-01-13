# MLFlow Experiment Tracker for Classification Models

This repository demonstrates the use of [MLFlow](https://mlflow.org/) for tracking machine learning experiments with a variety of classification algorithms. The dataset used is assumed to be a CSV file named `clients.csv`, which contains features for predicting `default_payment_next_month`. The workflow involves preprocessing, feature selection, model training, evaluation, and logging results to MLFlow.

## Features

- **Classification Models**: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Naive Bayes, SVM, and AdaBoost.
- **Experiment Tracking**: Logs hyperparameters, metrics, artifacts, and models to MLFlow.
- **Feature Selection**: Uses `SelectKBest` to select the top 50 features.
- **Custom Evaluation Metrics**: Accuracy, Precision, Recall, and F1 Score.
- **Dataset Splitting**: Splits the data into training (60%) and testing (40%) sets.

## Prerequisites

- Python 3.6+
- MLFlow installed and running locally
- The following Python libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `mlflow`

## Setting the tracking server on a local terminal:
```bash
conda activate PythonProject 
```
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifatcs --host 127.0.0.1 --port 5000 
```
