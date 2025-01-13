# Machine Learning Model Training and Experiment Logging with MLflow

## Overview

This repository contains a Python script that utilizes various machine learning algorithms to predict client default payments based on features contained in a dataset. The models are trained using scikit-learn, and MLflow is used to manage experiments, logging metrics, parameters, and model artifacts systematically.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- MLflow

For setting the tracking server on the local terminal 
```bash

conda activate PythonProject 

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifatcs --host 127.0.0.1 --port 5000 

and then set "http://127.0.0.1:5000" on the set_tracking_uri
