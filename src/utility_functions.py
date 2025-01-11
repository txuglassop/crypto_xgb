from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def classification_summary(y_pred: np.array, y_true: np.array):
    """
    Prints a summary of classification results.

    Params:
        np.array: y_pred - predictions on y_test
        np.array: y_true - actual test data, i.e. y_test

    returns:
        nothing 
    """
    print('\n------------ Classification Report ------------')
    print(classification_report(y_true, y_pred))

    print('\n\n-------------- Confusion Matrix --------------')
    print(confusion_matrix(y_true, y_pred))


