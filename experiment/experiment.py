import logging
from time import time

import numpy as np
import optuna
import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold

import data.data as data
from data import TabularDataFrame
from model import XGBoostClassifier

# from .optuna import OptimParam
from .utils import cal_metrics, set_seed

logger = logging.getLogger(__name__)

class ExpBase:
    def __init__(self, config):
        set_seed(config.seed)

        self.n_splits = config.n_splits
        self.model_name = config.model.name

        self.model_config = config.model.params
        self.exp_config = config.exp
        self.data_config = config.data

        dataframe: TabularDataFrame = getattr(data, self.data_config.name)(seed=config.seed, **self.data_config)
        dfs = dataframe.processed_dataframes()
        self.categories_dict = dataframe.get_categories_dict()
        self.train, self.test = dfs["train"], dfs["test"]
        self.columns = dataframe.all_columns
        self.target_column = dataframe.target_column
        self.label_encoder = dataframe.label_encoder

        self.input_dim = len(self.columns)
        self.output_dim = len(self.label_encoder.classes_)

        self.id = dataframe.id

        self.seed = config.seed
        self.init_writer()

    def init_writer(self):
        metrics = [
            "fold",
            "F1",
            "ACC",
            "AUC",
        ]
        self.writer = {m: [] for m in metrics}

    def add_results(self, i_fold, scores: dict, time):
        self.writer["fold"].append(i_fold)
        for m in self.writer.keys():
            if m == "fold":
                continue
            self.writer[m].append(scores[m])

    def each_fold(self, i_fold, train_data, val_data):
        x, y = self.get_x_y(train_data)

        model_config = self.get_model_config(i_fold=i_fold, x=x, y=y, val_data=val_data)
        model = XGBoostClassifier(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            verbose=self.exp_config.verbose,
        )
        start = time()
        model.fit(
            x,
            y,
            eval_set=(val_data[self.columns], val_data[self.target_column].values.squeeze()),
        )
        end = time() - start
        logger.info(f"[Fit {self.model_name}] Time: {end}")
        return model, end
    
    def run(self):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        y_test_pred_all = []
        for i_fold, (train_idx, val_idx) in enumerate(skf.split(self.train, self.train[self.target_column])):
            if len(self.writer["fold"]) != 0 and self.writer["fold"][-1] >= i_fold:
                logger.info(f"Skip {i_fold + 1} fold. Already finished.")
                continue

            train_data, val_data = self.train.iloc[train_idx], self.train.iloc[val_idx]
            model, time = self.each_fold(i_fold, train_data, val_data)

            score = cal_metrics(model, val_data, self.columns, self.target_column)
            score.update(model.evaluate(val_data[self.columns], val_data[self.target_column].values.squeeze()))
            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] val/ACC: {score['ACC']:.4f} | val/AUC: {score['AUC']:.4f} | "
                f"val/F1: {score['F1']}"
            )

            self.add_results(i_fold, score, time)

            y_test_pred_all.append(
                model.predict_proba(self.test[self.columns]).reshape(-1, 1, len(self.label_encoder.classes_))
            )

        y_test_pred_all = np.argmax(np.concatenate(y_test_pred_all, axis=1).mean(axis=1), axis=1)
        submit_df = pd.DataFrame(self.id)
        submit_df["Transported"] = self.label_encoder.inverse_transform(y_test_pred_all)
        print(submit_df)
        submit_df.to_csv("submit.csv", index=False)

    def get_model_config(self, *args, **kwargs):
        raise NotImplementedError()

    def get_x_y(self, train_data):
        x, y = train_data[self.columns], train_data[self.target_column].values.squeeze()
        return x, y
    

class ExpSimple(ExpBase):
    def __init__(self, config):
        super().__init__(config)

    def get_model_config(self, *args, **kwargs):
        return self.model_config