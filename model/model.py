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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

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

# 勾配ブースト分類器
class GBClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=42) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = GradientBoostingClassifier(
            verbose=self.verbose,
            random_state=seed,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns

        self.model.fit(X, y)

# ランダムフォレスト分類器の拡張版
class ETClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=42) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = ExtraTreesClassifier(
            verbose=self.verbose,
            random_state=seed,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns

        self.model.fit(X, y)

# k 近傍法分類器
class KNClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=42) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = KNeighborsClassifier(
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns

        self.model.fit(X, y)