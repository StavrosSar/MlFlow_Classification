import warnings
import logging
import numpy as np
import mlflow.sklearn
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics1(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='weighted')
    f1 = f1_score(actual, pred, average='weighted')
    metrics_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    return accuracy, precision, recall, f1, metrics_dict

def log_experiment(mlflow, run_name, model, params, metrics, artifact_location):
    """
    Log an MLflow experiment run with parameters, metrics, artifacts, and a model.

    Parameters:
    - mlflow: An MLflow client instance.
    - run_name: A string name to identify the run.
    - model: The trained model to log.
    - params: A dictionary of parameters to log.
    - metrics: A dictionary of metrics to log.
    - artifact_location: Directory containing artifacts to log.

    Returns: None
    """
    mlflow.start_run(run_name=run_name)
    mlflow.autolog(log_input_examples=True)
    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    # Uncomment the below if you need but params and metrics are automatically logged above on mlflow.autolog
    '''
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    '''
    mlflow.log_artifacts("data/")

    # Log the model
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    data = pd.read_csv("clients.csv")
    os.mkdir("data/")
    data.to_csv("data/clients.csv", index=False)

    train, test = train_test_split(data, test_size=0.4, random_state=42)

    # Data Preprocessing
    le = LabelEncoder()
    target = 'default_payment_next_month'
    train[target] = le.fit_transform(train[target])
    test[target] = le.fit_transform(test[target])

    # Storing the training and testing dataset
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    # Split
    train_x = train.drop([target], axis=1)
    test_x = test.drop([target], axis=1)
    train_y = train[target]
    test_y = test[target]

    # Feature Selection with SelectKBest
    k_best_features = 50
    selector = SelectKBest(f_classif, k=k_best_features)
    selector.fit(train_x, train_y)

    selected_features = train_x.columns[selector.get_support()]
    print("Selected features:", selected_features.tolist())
    input_features = selected_features.tolist()

    # Re-define train_x and test_x with the selected features
    train_x = train_x[input_features]
    test_x = test_x[input_features]

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # LogisticRegression
    if True:
        ###########First Experiment LogisticRegression   #############
        mlflow.set_experiment(experiment_name="LogisticRegression")

        # Hyperparameters
        C = 1.0
        penality = 'l1'
        model = LogisticRegression(penalty=penality, solver='liblinear')
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("Logistic Regression model (penality={}, C={}):".format(penality, C))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)


        log_experiment(mlflow, run_name="run1.1", model=model,
                       params={"penality": penality, "C": C}, metrics=metrics_dict, artifact_location="data/")


        ###########Second Experiment LogisticRegression   #############
        mlflow.set_experiment(experiment_name="LogisticRegression")

        penality = 'l2'
        model = LogisticRegression(penalty=penality, solver='liblinear')
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("Logistic Regression model (penality={}, C={}):".format(penality, C))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run1.2", model=model,
                       params={"penality": penality, "C": C}, metrics=metrics_dict, artifact_location="data/")


        ###########Third Experiment LogisticRegression   #############
        mlflow.set_experiment(experiment_name="LogisticRegression")

        penality = 'elasticnet'
        model = LogisticRegression(penalty=penality, solver='saga', l1_ratio=0.01)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("Logistic Regression model (penality={}, C={}):".format(penality, C))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run1.3", model=model,
                       params={"penality": penality, "C": C}, metrics=metrics_dict, artifact_location="data/")

    # DecisionTreeClassifier
    if True:
        ###########First Experiment DecisionTreeClassifier   #############
        mlflow.set_experiment(experiment_name="DecisionTreeClassifier")

        criterion = 'gini'
        model = DecisionTreeClassifier(criterion=criterion)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("DecisionTreeClassifier model (criterion={}):".format(criterion))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run2.1", model=model,
                       params={"criterion": criterion}, metrics=metrics_dict, artifact_location="data/")


        ###########Second Experiment DecisionTreeClassifier   #############
        mlflow.set_experiment(experiment_name="DecisionTreeClassifier")

        criterion = 'entropy'
        model = DecisionTreeClassifier(criterion=criterion)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("DecisionTreeClassifier model (criterion={}):".format(criterion))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run2.2", model=model,
                       params={"criterion": criterion}, metrics=metrics_dict, artifact_location="data/")

        ###########Third Experiment DecisionTreeClassifier   #############
        mlflow.set_experiment(experiment_name="DecisionTreeClassifier")

        criterion = 'log_loss'
        model = DecisionTreeClassifier(criterion=criterion)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("DecisionTreeClassifier model (criterion={}):".format(criterion))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run2.3", model=model,
                       params={"criterion": criterion}, metrics=metrics_dict, artifact_location="data/")

    # RandomForestClassifier
    if True:
        ###########First Experiment RandomForestClassifier   #############
        mlflow.set_experiment(experiment_name="RandomForestClassifier")

        criterion = 'gini'
        model = RandomForestClassifier(criterion=criterion)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("RandomForestClassifier model (criterion={}):".format(criterion))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run3.1", model=model,
                       params={"criterion": criterion}, metrics=metrics_dict, artifact_location="data/")

        ###########Second Experiment RandomForestClassifier   #############
        mlflow.set_experiment(experiment_name="RandomForestClassifier")

        criterion = 'entropy'
        model = RandomForestClassifier(criterion=criterion)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("RandomForestClassifier model (criterion={}):".format(criterion))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run3.2", model=model,
                       params={"criterion": criterion}, metrics=metrics_dict, artifact_location="data/")

        ###########Third Experiment RandomForestClassifier   #############
        mlflow.set_experiment(experiment_name="RandomForestClassifier")

        criterion = 'log_loss'
        model = RandomForestClassifier(criterion=criterion)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("RandomForestClassifier model (criterion={}):".format(criterion))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run3.3", model=model,
                       params={"criterion": criterion}, metrics=metrics_dict, artifact_location="data/")

    # KNeighborsClassifier
    if True:
        ###########First Experiment KNeighborsClassifier   #############
        mlflow.set_experiment(experiment_name="KNeighborsClassifier")

        n_neighbors = 7
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("KNeighborsClassifier model (n_neighbors={}):".format(n_neighbors))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run4.1", model=model,
                       params={"n_neighbors": n_neighbors}, metrics=metrics_dict, artifact_location="data/")

        ###########Second Experiment KNeighborsClassifier   #############
        mlflow.set_experiment(experiment_name="KNeighborsClassifier")

        n_neighbors = 10
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("KNeighborsClassifier model (n_neighbors={}):".format(n_neighbors))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run4.2", model=model,
                       params={"n_neighbors": n_neighbors}, metrics=metrics_dict, artifact_location="data/")

        ###########Third Experiment KNeighborsClassifier   #############
        mlflow.set_experiment(experiment_name="KNeighborsClassifier")

        n_neighbors = 14
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("KNeighborsClassifier model (n_neighbors={}):".format(n_neighbors))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run4.3", model=model,
                       params={"n_neighbors": n_neighbors}, metrics=metrics_dict, artifact_location="data/")

    # GaussianNB
    if True:
        ###########First Experiment GaussianNB   #############
        mlflow.set_experiment(experiment_name="GaussianNB")

        model = GaussianNB()
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("GaussianNB model (n_neighbors={}):".format(n_neighbors))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run5.1", model=model,
                       params={}, metrics=metrics_dict, artifact_location="data/")

    # SVM
    if True:
        ###########First Experiment SVM   #############
        mlflow.set_experiment(experiment_name="SVM")

        kernel = 'rbf'
        model = SVC(kernel=kernel)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("SVM model (kernel={}):".format(kernel))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run6.1", model=model,
                       params={"kernel": kernel}, metrics=metrics_dict, artifact_location="data/")

        ###########Second Experiment SVM   #############
        mlflow.set_experiment(experiment_name="SVM")

        kernel = 'sigmoid'
        model = SVC(kernel=kernel)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("SVM model (kernel={}):".format(kernel))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run6.2", model=model,
                       params={"kernel": kernel}, metrics=metrics_dict, artifact_location="data/")

    # AdaBoost
    if True:
        ########### First Experiment AdaBoost #############
        mlflow.set_experiment(experiment_name="AdaBoost")

        n_estimators = 50
        model = AdaBoostClassifier(n_estimators=n_estimators)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("AdaBoost model (n_estimators={}):".format(n_estimators))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run7.1", model=model,
                       params={"n_estimators": n_estimators}, metrics=metrics_dict, artifact_location="data/")

        ########### Second Experiment AdaBoost #############
        mlflow.set_experiment(experiment_name="AdaBoost")

        n_estimators = 100
        model = AdaBoostClassifier(n_estimators=n_estimators)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("AdaBoost model (n_estimators={}):".format(n_estimators))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run7.2", model=model,
                       params={"n_estimators": n_estimators}, metrics=metrics_dict, artifact_location="data/")

        ########### Third Experiment AdaBoost #############
        mlflow.set_experiment(experiment_name="AdaBoost")

        n_estimators = 150
        model = AdaBoostClassifier(n_estimators=n_estimators)
        model.fit(train_x, train_y)
        predicted_classes = model.predict(test_x)
        accuracy, precision, recall, f1, metrics_dict = eval_metrics1(test_y, predicted_classes)

        print("AdaBoost model (n_estimators={}):".format(n_estimators))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        log_experiment(mlflow, run_name="run7.3", model=model,
                       params={"n_estimators": n_estimators}, metrics=metrics_dict, artifact_location="data/")

    run = mlflow.last_active_run()
    print("Recent Active run id is {}".format(run.info.run_id))
    print("Recent Active run name is {}".format(run.info.run_name))