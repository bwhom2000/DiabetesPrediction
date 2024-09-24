import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
import mlflow


def train_logistic_regression(X_train, y_train, random_seed):
    """
    Train a logistic regression model.

    Parameters:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        random_seed (int): Seed for random number generation.

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    model = LogisticRegression(random_state=random_seed)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, params):
    """
    Train a decision tree classifier.

    Parameters:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        params (dict): Parameters for the decision tree classifier.

    Returns:
        DecisionTreeClassifier: The trained decision tree model.
    """
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    return model


def get_dt_params():
    """
    Generate a list of hyperparameter combinations for decision tree training.

    Returns:
        list: List of parameter dictionaries for decision tree classifiers.
    """
    max_depths = [None, 10, 20, 30]
    min_samples_splits = [2, 5, 10]
    min_samples_leaves = [1, 2, 4]

    dt_params = []

    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            for min_samples_leaf in min_samples_leaves:
                dt_params.append({
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                })

    return dt_params


def train_random_forest(X_train, y_train, params):
    """
    Train a random forest classifier.

    Parameters:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        params (dict): Parameters for the random forest classifier.

    Returns:
        RandomForestClassifier: The trained random forest model.
    """
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def get_rf_params():
    """
    Generate a list of hyperparameter combinations for random forest training.

    Returns:
        list: List of parameter dictionaries for random forest classifiers.
    """
    n_estimators = [50, 100, 150]
    max_depths = [10, 20]
    min_samples_splits = [2, 5, 10]

    rf_params = []

    for n in n_estimators:
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                rf_params.append({
                    'n_estimators': n,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                })

    return rf_params


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a model using accuracy, precision, and recall.

    Parameters:
        model: The trained model to evaluate.
        X_test (pd.DataFrame): Test feature data.
        y_test (pd.Series): True labels for the test data.

    Returns:
        tuple: A tuple containing accuracy, precision, and recall scores.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return accuracy, precision, recall


def log_model(model, recall, params):
    """
    Log the trained model and its evaluation metrics to MLflow.

    Parameters:
        model: The trained model to log.
        recall (float): The recall score of the model.
        params (dict): Parameters used to train the model.
    """
    # Start an MLflow run to log the model and metrics
    with mlflow.start_run():
        mlflow.log_param("best_params", params)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(model, "best_model_metaflow_kubernetes")
        mlflow.end_run()
