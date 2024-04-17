import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from .utils import f1_micro, f1_micro_lgb

class XGBoostClassifier:
    def __init__(self, input_dim, output_dim, verbose) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.verbose = verbose
        self.model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=self.output_dim,
            # eval_metric=f1_micro,
            early_stopping_rounds=50
        )

    def fit(self, X, y, eval_set):

        self.model.fit(X, y, eval_set=[eval_set], verbose=self.verbose > 0)
        pass
    
    def predict_proba(self, X):
        return self.model.predict_proba(X.values)

    def predict(self, X):
        return self.model.predict(X.values)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        results = {}
        results["ACC"] = accuracy_score(y, y_pred)
        y_score = self.predict_proba(X)[:,1]
        results["AUC"] = roc_auc_score(y, y_score)
        results["Precision"] = precision_score(y, y_pred, average="micro", zero_division=0)
        results["Recall"] = recall_score(y, y_pred, average="micro", zero_division=0)
        results["Specificity"] = recall_score(1 - y, 1 - y_pred, average="micro", zero_division=0)
        results["F1"] = f1_score(y, y_pred, average="micro", zero_division=0)
        return results
