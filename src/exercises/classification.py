import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.utils.config import RESOURCES_DIR, CLASSIFICATION_FILE
from src.utils.printers import print_df, print_classifier_results


# Computes the accuracy of the model using the confusion matrix
def compute_accuracy(cm):
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

    return accuracy


# Computes the precision of the model using the confusion matrix
def compute_precision(cm):
    precision = cm[0][0] / (cm[0][0] + cm[0][1])

    return precision


# Computes the recall of the model using the confusion matrix
def compute_recall(cm):
    recall = cm[0][0] / (cm[0][0] + cm[1][0])

    return recall


# Computes the F1 score of the model using the precision and recall
def compute_f1_score(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


# Evaluates the performance of a classifier by manually computing stuff...
def evaluate_classifier_dumb(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm[0][0], cm[1][1] = cm[1][1], cm[0][0]

    accuracy = compute_accuracy(cm)
    precision = compute_precision(cm)
    recall = compute_recall(cm)
    f1 = compute_f1_score(precision, recall)

    return accuracy, precision, recall, f1


# Evaluates the performance of a classifier using code written by others...
def evaluate_classifier_smart(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    return accuracy, precision, recall, f1


# Evaluates the performance of a classifier
def evaluate_classifier(y_test, y_pred, mode='dumb'):
    if mode == 'dumb':
        accuracy, precision, recall, f1 = evaluate_classifier_dumb(y_test, y_pred)
    elif mode == 'smart':
        accuracy, precision, recall, f1 = evaluate_classifier_smart(y_test, y_pred)
    else:
        return None, None, None, None

    return accuracy, precision, recall, f1


# Performs a classification on a sample dataset
def perform_classification():
    classification_dataset = CLASSIFICATION_FILE
    # Load the dataset from the resources folder
    dataset = pd.read_csv(RESOURCES_DIR + "/" + classification_dataset)

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

    # Build a classifier (less important: a decision tree will be used)
    clf = DecisionTreeClassifier(random_state=42)

    # Fit data from the training set to the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model using y_pred and y_test and print the results
    mode = 'dumb'
    accuracy, precision, recall, f1 = evaluate_classifier(y_test, y_pred, mode)
    print_classifier_results(accuracy, precision, recall, f1, mode)

    mode = 'smart'
    accuracy, precision, recall, f1 = evaluate_classifier(y_test, y_pred, mode)
    print_classifier_results(accuracy, precision, recall, f1, mode)
