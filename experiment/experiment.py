import logging
from time import time

import numpy as np
import optuna
import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold, train_test_split

import data.data as data
from data import TabularDataFrame
from model import get_classifier

from .optuna import OptimParam
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
        model = get_classifier(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
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
        self.train.to_csv("train.csv")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        y_test_pred_all = []
        score_all = 0
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

            score_all += score["ACC"]

            self.add_results(i_fold, score, time)

            y_test_pred_all.append(
                model.predict_proba(self.test[self.columns]).reshape(-1, 1, len(self.label_encoder.classes_))
            )
        
        y_test_pred_all = np.argmax(np.concatenate(y_test_pred_all, axis=1).mean(axis=1), axis=1)
        submit_df = pd.DataFrame(self.id)
        submit_df["Transported"] = self.label_encoder.inverse_transform(y_test_pred_all)
        print(submit_df)
        submit_df.to_csv("submit.csv", index=False)

        logger.info(f" {self.model_name} score average: {score_all/self.n_splits} ")

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

class ExpOptuna(ExpBase):
    def __init__(self, config):
        super().__init__(config)
        self.n_trials = config.exp.n_trials
        self.n_startup_trials = config.exp.n_startup_trials

        self.storage = config.exp.storage
        self.study_name = config.exp.study_name
        self.cv = config.exp.cv
        self.n_jobs = config.exp.n_jobs

    def run(self):
        if self.exp_config.delete_study:
            for i in range(self.n_splits):
                optuna.delete_study(
                    study_name=f"{self.exp_config.study_name}_{i}",
                    storage=f"sqlite:///{to_absolute_path(self.exp_config.storage)}/optuna.db",
                )
                print(f"delete successful in {i}")
            return
        super().run()
        # train_data, val_data = train_test_split(self.train, test_size=0.2)
        # x, y = self.get_x_y(self.train)
        # print(self.get_model_config(1, x, y))

    def get_model_config(self, i_fold, x, y, val_data):
        op = OptimParam(
            self.model_name,
            default_config=self.model_config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            X=x,
            y=y,
            val_data=val_data,
            columns=self.columns,
            target_column=self.target_column,
            n_trials=self.n_trials,
            n_startup_trials=self.n_startup_trials,
            storage=self.storage,
            study_name=f"{self.study_name}_{i_fold}",
            # study_name=f"{self.study_name}",
            cv=self.cv,
            n_jobs=self.n_jobs,
            seed=self.seed,
        )
        return op.get_best_config()

class ExpStacking(ExpBase):
    def __init__(self, config):
        super().__init__(config)
    
    def run(self):
        val_xgb = None
        test_xgb = None

        val_lgbm = None
        test_lgbm = None

        val_cat = None
        test_cat = None

        val_rf = None
        test_rf = None

        val_gb = None
        test_gb = None

        val_et = None
        test_et = None

        val_kn = None
        test_kn = None

        predict_columns = []

        if self.exp_config.xgboost:
            self.model_name = 'xgboost'
            logger.info(f"Stacking {self.model_name} predict proba start! ")
            val_xgb , test_xgb = self.stacking_predict()
            predict_columns.append('xgb_False')
            predict_columns.append('xgb_True')
        if self.exp_config.lightgbm:
            self.model_name = 'lightgbm'
            logger.info(f"Stacking {self.model_name} predict proba start! ")
            val_lgbm, test_lgbm = self.stacking_predict()
            predict_columns.append('lgbm_False')
            predict_columns.append('lgbm_True')
        if self.exp_config.catboost:
            self.model_name = 'catboost'
            logger.info(f"Stacking {self.model_name} predict proba start! ")
            val_cat, test_cat = self.stacking_predict()
            predict_columns.append('cat_False')
            predict_columns.append('cat_True')
        if self.exp_config.randomforest:
            self.model_name = 'randomforest'
            logger.info(f"Stacking {self.model_name} predict proba start! ")
            val_rf, test_rf = self.stacking_predict()
            predict_columns.append('rf_False')
            predict_columns.append('rf_True')
        if self.exp_config.gradientboosting:
            self.model_name = 'gradientboosting'
            logger.info(f"Stacking {self.model_name} predict proba start! ")
            val_gb, test_gb = self.stacking_predict()
            predict_columns.append('gb_False')
            predict_columns.append('gb_True')
        if self.exp_config.extratrees:
            self.model_name = 'extratrees'
            logger.info(f"Stacking {self.model_name} predict proba start! ")
            val_et, test_et = self.stacking_predict()
            predict_columns.append('et_False')
            predict_columns.append('et_True')
        if self.exp_config.kneighbors:
            self.model_name = 'kneighbors'
            logger.info(f"Stacking {self.model_name} predict proba start! ")
            val_kn, test_kn = self.stacking_predict()
            predict_columns.append('kn_False')
            predict_columns.append('kn_True')
        
        train_predict = np.concatenate([val for val in [val_xgb, val_lgbm, val_cat, val_rf, val_gb, val_et, val_kn] if val is not None], axis=1)
        test_predict = np.concatenate([test for test in [test_xgb, test_lgbm, test_cat, test_rf, test_gb, test_et, test_kn] if test is not None], axis=1)

        train_predict = pd.DataFrame(train_predict, columns=predict_columns)
        train_predict = pd.concat([train_predict, self.train[self.target_column]], axis=1)

        test_predict = pd.DataFrame(test_predict, columns=predict_columns)

        if self.exp_config.drop_False:
            # 条件に合致するカラムを抽出
            columns_to_drop = [col for col in predict_columns if col.endswith("_False")]
            # カラムを削除
            train_predict.drop(columns_to_drop, axis=1, inplace=True)
            test_predict.drop(columns_to_drop, axis=1, inplace=True)

        # predictの中身を見る場合は以下を実行
        train_predict.to_csv("train_predict.csv", index=False)
        test_predict.to_csv("test_predict.csv", index=False)

        # model_nameを2層目のモデル名に変更
        self.model_name = 'xgboost2'
        # カラム名を2層目用に変更
        self.columns = test_predict.columns
        # print(self.columns)
        # exit()


        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        y_test_pred_all = []
        score_all = 0
        for i_fold, (train_idx, val_idx) in enumerate(skf.split(train_predict, train_predict[self.target_column])):

            train_data, val_data = train_predict.iloc[train_idx], train_predict.iloc[val_idx]
            model, time = self.each_fold(i_fold, train_data, val_data)

            score = cal_metrics(model, val_data, self.columns, self.target_column)
            score.update(model.evaluate(val_data[self.columns], val_data[self.target_column].values.squeeze()))
            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] val/ACC: {score['ACC']:.4f} | val/AUC: {score['AUC']:.4f} | "
                f"val/F1: {score['F1']}"
            )

            score_all += score["ACC"]

            self.add_results(i_fold, score, time)

            y_test_pred_all.append(
                model.predict_proba(test_predict[self.columns]).reshape(-1, 1, len(self.label_encoder.classes_))
            )
        
        y_test_pred_all = np.argmax(np.concatenate(y_test_pred_all, axis=1).mean(axis=1), axis=1)
        submit_df = pd.DataFrame(self.id)
        submit_df["Transported"] = self.label_encoder.inverse_transform(y_test_pred_all)
        print(submit_df)
        submit_df.to_csv("submit.csv", index=False)

        logger.info(f" {self.model_name} score average: {score_all/self.n_splits} ")




    def stacking_predict(self):
        val_predict = np.zeros((self.train[self.columns].shape[0], self.output_dim))
        test_predict = np.zeros((self.test[self.columns].shape[0], self.output_dim))
        test_skf_predict = np.zeros((self.n_splits, self.test[self.columns].shape[0], self.output_dim))

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        y_test_pred_all = []
        score_all = 0
        for i_fold, (train_idx, val_idx) in enumerate(skf.split(self.train, self.train[self.target_column])):

            train_data, val_data = self.train.iloc[train_idx], self.train.iloc[val_idx]
            model, time = self.each_fold(i_fold, train_data, val_data)

            val_predict[val_idx] = model.predict_proba(val_data[self.columns])
            test_skf_predict[i_fold, :] = model.predict_proba(self.test[self.columns])

            score = cal_metrics(model, val_data, self.columns, self.target_column)
            score.update(model.evaluate(val_data[self.columns], val_data[self.target_column].values.squeeze()))
            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] val/ACC: {score['ACC']:.4f} | val/AUC: {score['AUC']:.4f} | "
                f"val/F1: {score['F1']}"
            )

            score_all += score["ACC"]

            self.add_results(i_fold, score, time)

            y_test_pred_all.append(
                model.predict_proba(self.test[self.columns]).reshape(-1, 1, len(self.label_encoder.classes_))
            )
        
        test_predict[:] = test_skf_predict.mean(axis=0)

        logger.info(f" {self.model_name} score average: {score_all/self.n_splits} ")

        return val_predict, test_predict
        # y_test_pred_all = np.argmax(np.concatenate(y_test_pred_all, axis=1).mean(axis=1), axis=1)


        # submit_df = pd.DataFrame(self.id)
        # submit_df["Transported"] = self.label_encoder.inverse_transform(y_test_pred_all)
        # print(submit_df)
        # submit_df.to_csv("submit.csv", index=False)

        # print('score_average', score_all/self.n_splits)

    def get_model_config(self, *args, **kwargs):
        if self.model_name == 'xgboost':
            return self.model_config.xgboost
        elif self.model_name == 'lightgbm':
            return self.model_config.lightgbm
        elif self.model_name == 'catboost':
            return self.model_config.catboost
        elif self.model_name == 'randomforest':
            return self.model_config.randomforest
        elif self.model_name == 'gradientboosting':
            return self.model_config.gradientboosting
        elif self.model_name == 'extratrees':
            return self.model_config.extratrees
        elif self.model_name == 'kneighbors':
            return self.model_config.kneighbors
        # 2層目のモデル
        elif self.model_name == 'xgboost2':
            return self.model_config.xgboost2
            
        