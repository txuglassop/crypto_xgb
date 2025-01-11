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

def get_jump_lookup(num_classes):
    if num_classes == 3:
        jump_lookup = {
            'down':0,
            'neutral':1,
            'up':2
        }
        return jump_lookup
    elif num_classes == 5:
        jump_lookup = {
            'big_down':0,
            'small_down':1,
            'neutral':2,
            'small_up':3,
            'big_up':4
        }
    try:
        return jump_lookup
    except:
        print(f'Could not find a lookup for {num_classes} classes.')
        raise SystemError