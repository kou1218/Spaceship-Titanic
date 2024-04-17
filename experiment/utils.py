"""
This module contains several functions that are used in various stages of the process
"""
import os
import pickle
import random

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_object(obj, output_path: str):
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def load_object(input_path: str):
    with open(input_path, "rb") as f:
        return pickle.load(f)


def cal_auc_score(model, data, feature_cols, label_col):
    pred_proba = model.predict_proba(data[feature_cols])[:,1]
    auc = roc_auc_score(data[label_col].values.tolist(), pred_proba)
    return auc


def cal_acc_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    acc = accuracy_score(data[label_col], pred)
    return acc


def cal_metrics(model, data, feature_cols, label_col):
    acc = cal_acc_score(model, data, feature_cols, label_col)
    auc = cal_auc_score(model, data, feature_cols, label_col)
    return {"ACC": acc, "AUC": auc}