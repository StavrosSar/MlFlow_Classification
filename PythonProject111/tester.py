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

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='weighted')
    f1 = f1_score(actual, pred, average='weighted')
    return accuracy, precision, recall, f1

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

    mlflow.set_tracking_uri(uri="")

    print("The set tracking uri is ", mlflow.get_tracking_uri())

    #LogisticRegression
    if True:
        C = 1.0

        ###########First Experiment LogisticRegression   #############
        print("First Experiment LogisticRegression ")
        exp = mlflow.set_experiment(experiment_name="LogisticRegression")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run1.1", experiment_id=exp.experiment_id)

        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        penality = 'l1'
        dt = LogisticRegression(penalty=penality, solver='liblinear')
        dt.fit(train_x, train_y)

        predicted_classes = dt.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"Logistic Regression model (penality={penality}, C={C}):")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "penality": penality,
            "C": C
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(dt, "model")

        mlflow.end_run()

        ###########First Experiment LogisticRegression   #############
        print("Second Experiment LogisticRegression ")
        exp = mlflow.set_experiment(experiment_name="LogisticRegression")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run1.2", experiment_id=exp.experiment_id)
            
        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        penality = 'l2'
        dt = LogisticRegression(penalty=penality)
        dt.fit(train_x, train_y)

        predicted_classes = dt.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"Logistic Regression model (penality={penality}, C={C}):")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "penality": penality,
            "C": C
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(dt, "model")

        mlflow.end_run()

        ###########First Experiment LogisticRegression   #############
        print("Third Experiment LogisticRegression ")
        exp = mlflow.set_experiment(experiment_name="LogisticRegression")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run1.3", experiment_id=exp.experiment_id)

        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        penality = 'elasticnet'
        dt = LogisticRegression(penalty=penality, solver='saga', l1_ratio=0.5)
        dt.fit(train_x, train_y)

        predicted_classes = dt.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"Logistic Regression model (penality={penality}, C={C}):")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "penality": penality,
            "C": C
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(dt, "model")

        mlflow.end_run()

    #DecisionTreeClassifier
    if True:
        ###########Second Experiment DecisionTreeClassifier   #############
        print("First Experiment DecisionTreeClassifier ")
        exp = mlflow.set_experiment(experiment_name="DecisionTreeClassifier")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run2.1", experiment_id=exp.experiment_id)
       
        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        criterion = 'gini'
        dt = DecisionTreeClassifier(criterion = criterion)
        dt.fit(train_x, train_y)

        predicted_classes = dt.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"DecisionTreeClassifier model (criterion={criterion}")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "criterion": criterion,
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(dt, "model")

        mlflow.end_run()

        ###########Second Experiment DecisionTreeClassifier   #############
        print("First Experiment DecisionTreeClassifier ")
        exp = mlflow.set_experiment(experiment_name="DecisionTreeClassifier")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run2.2", experiment_id=exp.experiment_id)

        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        criterion = 'entropy'
        dt = DecisionTreeClassifier(criterion=criterion)
        dt.fit(train_x, train_y)

        predicted_classes = dt.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"DecisionTreeClassifier model (criterion={criterion}")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "criterion": criterion,
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(dt, "model")

        mlflow.end_run()

        ###########Second Experiment DecisionTreeClassifier   #############
        print("Third Experiment DecisionTreeClassifier ")
        exp = mlflow.set_experiment(experiment_name="DecisionTreeClassifier")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run2.3", experiment_id=exp.experiment_id)
      
        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        criterion = 'log_loss'
        dt = DecisionTreeClassifier(criterion=criterion)
        dt.fit(train_x, train_y)

        predicted_classes = dt.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"DecisionTreeClassifier model (criterion={criterion}")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "criterion": criterion,
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(dt, "model")

        mlflow.end_run()

    #RandomForestClassifier
    if True:
            ###########Third Experiment RandomForestClassifier   #############
            print("First Experiment RandomForestClassifier ")
            exp = mlflow.set_experiment(experiment_name="RandomForestClassifier")

            print("Name: {}".format(exp.name))
            print("Experiment_id: {}".format(exp.experiment_id))
            print("Artifact Location: {}".format(exp.artifact_location))
            print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
            print("Creation timestamp: {}".format(exp.creation_time))

            mlflow.start_run(run_name="run3.1", experiment_id=exp.experiment_id)
            
            mlflow.autolog(log_input_examples=True)
            current_run = mlflow.active_run()
            print("Active run id is {}".format(current_run.info.run_id))
            print("Active run name is {}".format(current_run.info.run_name))

            # Hyperparameters
            criterion = 'gini'
            dt = RandomForestClassifier(criterion=criterion)
            dt.fit(train_x, train_y)

            predicted_classes = dt.predict(test_x)

            (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

            print(f"RandomForestClassifier model (criterion={criterion}")
            print("  Accuracy: %s" % accuracy)
            print("  Precision: %s" % precision)
            print("  Recall: %s" % recall)
            print("  F1 Score: %s" % f1)

            # Logging parameters
            params = {
                "criterion": criterion,
            }
            mlflow.log_params(params)

            # Logging Metrics
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            mlflow.log_metrics(metrics)

            # Logging artifacts the data
            mlflow.log_artifacts("data/")
            # Logging model
            #mlflow.sklearn.log_model(dt, "model")

            mlflow.end_run()

            ###########Third Experiment RandomForestClassifier   #############
            print("Second Experiment RandomForestClassifier ")
            exp = mlflow.set_experiment(experiment_name="RandomForestClassifier")

            print("Name: {}".format(exp.name))
            print("Experiment_id: {}".format(exp.experiment_id))
            print("Artifact Location: {}".format(exp.artifact_location))
            print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
            print("Creation timestamp: {}".format(exp.creation_time))

            mlflow.start_run(run_name="run3.2", experiment_id=exp.experiment_id)

            mlflow.autolog(log_input_examples=True)
            current_run = mlflow.active_run()
            print("Active run id is {}".format(current_run.info.run_id))
            print("Active run name is {}".format(current_run.info.run_name))

            # Hyperparameters
            criterion = 'entropy'
            dt = RandomForestClassifier(criterion=criterion)
            dt.fit(train_x, train_y)

            predicted_classes = dt.predict(test_x)

            (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

            print(f"RandomForestClassifier model (criterion={criterion}")
            print("  Accuracy: %s" % accuracy)
            print("  Precision: %s" % precision)
            print("  Recall: %s" % recall)
            print("  F1 Score: %s" % f1)

            # Logging parameters
            params = {
                "criterion": criterion,
            }
            mlflow.log_params(params)

            # Logging Metrics
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            mlflow.log_metrics(metrics)

            # Logging artifacts the data
            mlflow.log_artifacts("data/")
            # Logging model
            #mlflow.sklearn.log_model(dt, "model")

            mlflow.end_run()

            ###########Third Experiment RandomForestClassifier   #############
            print("Third Experiment RandomForestClassifier ")
            exp = mlflow.set_experiment(experiment_name="RandomForestClassifier")

            print("Name: {}".format(exp.name))
            print("Experiment_id: {}".format(exp.experiment_id))
            print("Artifact Location: {}".format(exp.artifact_location))
            print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
            print("Creation timestamp: {}".format(exp.creation_time))

            mlflow.start_run(run_name="run3.3", experiment_id=exp.experiment_id)

            mlflow.autolog(log_input_examples=True)
            current_run = mlflow.active_run()
            print("Active run id is {}".format(current_run.info.run_id))
            print("Active run name is {}".format(current_run.info.run_name))

            # Hyperparameters
            criterion = 'log_loss'
            dt = RandomForestClassifier(criterion=criterion)
            dt.fit(train_x, train_y)

            predicted_classes = dt.predict(test_x)

            (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

            print(f"RandomForestClassifier model (criterion={criterion}")
            print("  Accuracy: %s" % accuracy)
            print("  Precision: %s" % precision)
            print("  Recall: %s" % recall)
            print("  F1 Score: %s" % f1)

            # Logging parameters
            params = {
                "criterion": criterion,
            }
            mlflow.log_params(params)

            # Logging Metrics
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            mlflow.log_metrics(metrics)

            # Logging artifacts the data
            mlflow.log_artifacts("data/")
            # Logging model
            #mlflow.sklearn.log_model(dt, "model")

            mlflow.end_run()

    #KNeighborsClassifier
    if True:
        ###########Fourth Experiment KNeighborsClassifier   #############
        print("First Experiment KNeighborsClassifier ")
        exp = mlflow.set_experiment(experiment_name="KNeighborsClassifier")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run4.1", experiment_id=exp.experiment_id)

        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        n_neighbors = 7
        dt = KNeighborsClassifier(n_neighbors=n_neighbors)
        dt.fit(train_x, train_y)

        predicted_classes = dt.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"KNeighborsClassifier model (n_neighbors={n_neighbors}")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "n_neighbors": n_neighbors,
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(dt, "model")

        mlflow.end_run()

        ###########Fourth Experiment KNeighborsClassifier   #############
        print("Second Experiment KNeighborsClassifier ")
        exp = mlflow.set_experiment(experiment_name="KNeighborsClassifier")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run4.2", experiment_id=exp.experiment_id)

        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        n_neighbors = 10
        dt = KNeighborsClassifier(n_neighbors=n_neighbors)
        dt.fit(train_x, train_y)

        predicted_classes = dt.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"KNeighborsClassifier model (n_neighbors={n_neighbors}")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "n_neighbors": n_neighbors,
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(dt, "model")

        mlflow.end_run()

        ###########Fourth Experiment KNeighborsClassifier   #############
        print("Third Experiment KNeighborsClassifier ")
        exp = mlflow.set_experiment(experiment_name="KNeighborsClassifier")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run4.3", experiment_id=exp.experiment_id)

        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        n_neighbors = 14
        dt = KNeighborsClassifier(n_neighbors=n_neighbors)
        dt.fit(train_x, train_y)

        predicted_classes = dt.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"KNeighborsClassifier model (n_neighbors={n_neighbors}")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "n_neighbors": n_neighbors,
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(dt, "model")

        mlflow.end_run()

    #GaussianNB
    if True:
        ###########Fifth Experiment GaussianNB   #############
        print("Experiment GaussianNB ")
        exp = mlflow.set_experiment(experiment_name="GaussianNB")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run5.1", experiment_id=exp.experiment_id)

        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))


        dt = GaussianNB()
        dt.fit(train_x, train_y)

        predicted_classes = dt.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f" GaussianNB model ")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(dt, "model")

        mlflow.end_run()
    
    #SVM
    if True:
        ###########Sixth Experiment SVM   #############
        print("First Experiment SVM ")
        exp = mlflow.set_experiment(experiment_name="SVM")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run6.1", experiment_id=exp.experiment_id)

        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        kernel = 'rbf'
        dt = SVC(kernel=kernel)
        dt.fit(train_x, train_y)

        predicted_classes = dt.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"SVM model (kernel={kernel}")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "kernel": kernel,
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(dt, "model")

        mlflow.end_run()

        ###########Sitxh Experiment SVM   #############
        print("Second Experiment SVM ")
        exp = mlflow.set_experiment(experiment_name="SVM")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run6.2", experiment_id=exp.experiment_id)
        
        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        kernel = 'sigmoid'
        dt = SVC(kernel=kernel)
        dt.fit(train_x, train_y)

        predicted_classes = dt.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"SVM model (kernel={kernel}")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "kernel": kernel,
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(dt, "model")

        mlflow.end_run()

    #AdaBoost
    if True:
        ########### Seventh Experiment AdaBoost #############
        print("First Experiment AdaBoost ")
        exp = mlflow.set_experiment(experiment_name="AdaBoost")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run7.1", experiment_id=exp.experiment_id)

        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        n_estimators = 50  # You can adjust the n_estimators parameter
        ab_classifier = AdaBoostClassifier(n_estimators=n_estimators)
        ab_classifier.fit(train_x, train_y)

        predicted_classes = ab_classifier.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"AdaBoostClassifier model (n_estimators={n_estimators})")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(ab_classifier, "model")

        mlflow.end_run()

        ########### Seventh Experiment AdaBoost #############
        print("Second Experiment AdaBoost ")
        exp = mlflow.set_experiment(experiment_name="AdaBoost")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run7.2", experiment_id=exp.experiment_id)

        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        n_estimators = 100  # Vary n_estimators for the second experiment
        ab_classifier = AdaBoostClassifier(n_estimators=n_estimators)
        ab_classifier.fit(train_x, train_y)

        predicted_classes = ab_classifier.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"AdaBoostClassifier model (n_estimators={n_estimators})")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "n_estimators": n_estimators,
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(ab_classifier, "model")

        mlflow.end_run()

        ########### Seventh Experiment AdaBoost #############
        print("Third Experiment AdaBoost ")
        exp = mlflow.set_experiment(experiment_name="AdaBoost")

        print("Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))

        mlflow.start_run(run_name="run7.3", experiment_id=exp.experiment_id)

        mlflow.autolog(log_input_examples=True)
        current_run = mlflow.active_run()
        print("Active run id is {}".format(current_run.info.run_id))
        print("Active run name is {}".format(current_run.info.run_name))

        # Hyperparameters
        n_estimators = 150  # You can vary n_estimators for the third experiment
        ab_classifier = AdaBoostClassifier(n_estimators=n_estimators)
        ab_classifier.fit(train_x, train_y)

        predicted_classes = ab_classifier.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"AdaBoostClassifier model (n_estimators={n_estimators})")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "n_estimators": n_estimators,
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        #mlflow.sklearn.log_model(ab_classifier, "model")

        mlflow.end_run()


    run = mlflow.last_active_run()
    print("Recent Active run id is {}".format(run.info.run_id))
    print("Recent Active run name is {}".format(run.info.run_name))