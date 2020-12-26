import pandas as pd


# Pretty prints a data frame without display limits
def print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


# Pretty prints the results of the classifier performance evaluation
def print_classifier_results(accuracy, precision, recall, f1, mode):
    print()
    print('~~~~~~~~~~~~~~~~ CLASSIFICATION RESULTS (' + mode.upper() + ') ~~~~~~~~~~~~~~~~')
    print(' Accuracy: ' + str(accuracy))
    print('Precision: ' + str(precision))
    print('   Recall: ' + str(recall))
    print(' F1 Score: ' + str(f1))


# Pretty prints the results of the regressor performance evaluation
def print_regressor_results(rmse, mae, r_squared, mode):
    print()
    print('~~~~~~~~~~~~~~~~ REGRESSION RESULTS (' + mode.upper() + ') ~~~~~~~~~~~~~~~~')
    print('Root Mean Squared Error (RMSE): ' + str(rmse))
    print('     Mean Absolute Error (MAE): ' + str(mae))
    print('               R-squared Score: ' + str(r_squared))


# Pretty prints the results of the multiple classifications
def print_fitting_results(accuracy_list):

    print()
    print('~~~~~~~~~~~~~~~~ CLASSIFICATION RESULTS ~~~~~~~~~~~~~~~~')

    for accuracy_pair in accuracy_list:
        print('Training Set Accuracy 1: ' + str(accuracy_pair[0]))
        print('    Test Set Accuracy 1: ' + str(accuracy_pair[1]))
        print()


# Pretty prints the results of the clustering algorithm
def print_clustering_results(silhouette_score):

    print()
    print('~~~~~~~~~~~~~~~~ CLUSTERING RESULTS ~~~~~~~~~~~~~~~~')
    print('Silhouette Score: ' + str(silhouette_score))
