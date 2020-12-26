import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.exercises import classification
from src.utils.config import RESOURCES_DIR, CLASSIFICATION_FILE
from src.utils.printers import print_fitting_results, print_df


# Builds 3 different classifiers (less important: a decision tree will be used)
def train_some_models(X_train, y_train):
    return [DecisionTreeClassifier(max_depth=1, random_state=42).fit(X_train, y_train),
            DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train),
            DecisionTreeClassifier(max_depth=32, random_state=42).fit(X_train, y_train)]


# Makes prediction on both the training set and test set
# HINT 1: you can reuse the "evaluate_classifier" function from "classification.py"
# HINT 2: the models are already trained
def make_predictions(clf, X_train, X_test, y_train, y_test):
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_accuracy = classification.evaluate_classifier_smart(y_train, y_pred_train)[0]
    test_accuracy = classification.evaluate_classifier_smart(y_test, y_pred_test)[0]

    return train_accuracy, test_accuracy


"""
TODO - TASK B

First model - 73% train accuracy and 74% test accuracy - underfitting
Second model - 84% train accuracy and 79% test accuracy - best model
Third model - 100% train accuracy and 74% test accuracy - overfitting
"""


# Performs multiple classifications to understand the concept of data fitting
# noinspection PyTypeChecker
def perform_multiple_classifications():
    # Load the dataset from the resources folder
    dataset = pd.read_csv(RESOURCES_DIR + "/" + CLASSIFICATION_FILE)

    # Take a look at the first entries of the dataset
    print_df(dataset.head())

    # Extract the features from the dataset
    X = dataset.iloc[:, :-1]
    # print_df(X.head())

    # Extract the labels from the dataset
    y = dataset.iloc[:, -1:]
    # print_df(y.head())

    # Split the data into a training set and a test set with a 80-20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build 3 different models (they will be already trained)
    clf_list = train_some_models(X_train, y_train)

    # Make predictions on the training and test sets and evaluate the performance of each model
    accuracy_list = []
    for clf in clf_list:
        accuracy_list.append(make_predictions(clf, X_train, X_test, y_train, y_test))

    # Print the performance evaluation results
    print_fitting_results(accuracy_list)
