import numpy as np
from sklearn.metrics import f1_score


def f1_micro(y_true, y_pred):
    return -f1_score(y_true, y_pred, average="micro", zero_division=0)


def f1_micro_lgb(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return "f1_micro", f1_score(y_true, y_pred, average="micro", zero_division=0), True