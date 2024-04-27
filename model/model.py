import xgboost as xgb
import lightgbm as lgb
import catboost as cbt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from .utils import f1_micro, f1_micro_lgb
from .base_model import BaseClassifier

from sklearn.ensemble import RandomForestClassifier

class XGBoostClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=self.output_dim,
            # eval_metric=f1_micro,
            early_stopping_rounds=50,
            **self.model_config
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

class LightGBMClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=42) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = lgb.LGBMClassifier(
            objective="binary",
            verbose=self.verbose,
            random_state=seed,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns

        self.model.fit(
            X,
            y,
            eval_set=[eval_set],
            # eval_metric=f1_micro_lgb,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=self.verbose > 0)],
        )
    
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

class CBTClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=42) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = cbt.CatBoostClassifier(
            loss_function="Logloss",
            verbose=self.verbose,
            random_seed=seed,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns

        self.model.fit(
            X,
            y,
            eval_set=[eval_set],
            # eval_metric=f1_micro_lgb,
        )
    
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

class RFClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=42) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = RandomForestClassifier(
            verbose=self.verbose,
            random_state=seed,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns

        self.model.fit(X, y)
    
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