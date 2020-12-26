import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.utils.config import REGRESSION_FILE, RESOURCES_DIR
from src.utils.printers import print_df, print_regressor_results

CHUNKS = 100
MIN_CHUNK_SIZE = 1000
MAX_CHUNK_SIZE = 1000000


# Computes the RMSE of the model
def compute_rmse(y_test, y_pred):
    rmse = list(np.sqrt(np.sum((y_test - y_pred) ** 2) / len(y_test)))[0]
    return rmse


# Computes the MAE of the model
def compute_mae(y_test, y_pred):
    mae = list(np.sum(np.abs(y_pred - y_test)) / len(y_pred))[0]
    return mae


# Computes the R-squared score of the model
def compute_r2_score(y_test, y_pred):
    r2 = 1 - list(np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))[0]
    return r2


# Evaluates the performance of a regressor by manually computing stuff...
def evaluate_regressor_dumb(y_test, y_pred):
    rmse = compute_rmse(y_test, y_pred)
    mae = compute_mae(y_test, y_pred)
    r_squared = compute_r2_score(y_test, y_pred)

    return rmse, mae, r_squared


# Evaluates the performance of a regressor using code written by others...
def evaluate_regressor_smart(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    return rmse, mae, r_squared


# Evaluates the performance of a regressor
def evaluate_regressor(y_test, y_pred, mode='dumb'):
    if mode == 'dumb':
        rmse, mae, r_squared = evaluate_regressor_dumb(y_test, y_pred)
    elif mode == 'smart':
        rmse, mae, r_squared = evaluate_regressor_smart(y_test, y_pred)
    else:
        return None, None, None

    return rmse, mae, r_squared


# Plots the evolution of the RMSE when the dataset size varies
def plot_rmse_evolution(X, y, chunks, min_chunk_size, max_chunk_size):
    n = min(X.shape[0], max_chunk_size)
    chunk_size = int((n - min_chunk_size) / chunks)

    # Create 2 lists used in the plotting logic
    size_list = []
    rmse_list = []

    # Train a model for each chunk
    for i in range(0, chunks):
        # Compute the size of the current chunk
        size = min_chunk_size + (i + 1) * chunk_size
        size_list.append(size)

        # Select a chunk from the whole dataset
        sample_X = X.sample(n=size, random_state=42)
        sample_y = y.sample(n=size, random_state=42)

        # Split the data into a training set and a test set with a 80-20 ratio
        X_train, X_test, y_train, y_test = train_test_split(sample_X, sample_y, test_size=0.2, random_state=42)

        # Build a linear regressor
        regressor = LinearRegression()

        # Fit data from the training set to the regressor
        regressor.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = regressor.predict(X_test)

        # Compute the rmse
        rmse, _, _ = evaluate_regressor(y_test, y_pred, 'smart')
        rmse_list.append(rmse)

    plt.plot(size_list, rmse_list)
    plt.title('CHUNKS=%d, MIN_CHUNKS=%d, MAX_CHUNKS=%d' % (CHUNKS, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE))
    plt.ylabel('RMSE value')
    plt.xlabel('Data size')
    plt.show()


# Performs a regression on a sample dataset
def perform_regression():
    # Load the dataset from the resources folder
    dataset = pd.read_csv(RESOURCES_DIR + "/" + REGRESSION_FILE, dtype=np.float64)

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

    # Build a linear regressor
    regressor = LinearRegression()

    # Fit data from the training set to the regressor
    regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = regressor.predict(X_test)

    # Evaluate the model using y_pred and y_test and print the results
    mode = 'dumb'
    rmse, mae, r_squared = evaluate_regressor(y_test, y_pred, mode)
    print_regressor_results(rmse, mae, r_squared, mode)

    mode = 'smart'
    rmse, mae, r_squared = evaluate_regressor(y_test, y_pred, mode)
    print_regressor_results(rmse, mae, r_squared, mode)

    plot_rmse_evolution(X, y, CHUNKS, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)
