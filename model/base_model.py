
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaseClassifier:
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = model_config
        self.verbose = verbose

    def fit(self, X, y, eval_set):
        raise NotImplementedError()

    def predict_proba(self, X):
        return self.model.predict_proba(X.values)

    def predict(self, X):
        return self.model.predict(X.values)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        results = {}
        results["ACC"] = accuracy_score(y, y_pred)
        y_score = self.predict_proba(X)
        results["AUC"] = roc_auc_score(y, y_score, multi_class="ovr")
        results["Precision"] = precision_score(y, y_pred, average="micro", zero_division=0)
        results["Recall"] = recall_score(y, y_pred, average="micro", zero_division=0)
        results["Specificity"] = recall_score(1 - y, 1 - y_pred, average="micro", zero_division=0)
        results["F1"] = f1_score(y, y_pred, average="micro", zero_division=0)
        return results
