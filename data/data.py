import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.impute import SimpleImputer

from .utils import feature_name_combiner

logger = logging.getLogger(__name__)


# Copied from https://github.com/pfnet-research/deep-table.
# Modified by somaonishi and shoyameguro.
class TabularDataFrame(object):
    columns = [
        'PassengerId',
        'HomePlanet',
        'CryoSleep',
        'Cabin',
        'Destination',
        'Age',
        'VIP',
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck',
        'Name'
    ]
    continuous_columns = []
    categorical_columns = []
    binary_columns = []
    target_column = "Transported"

    def __init__(
        self,
        seed,
        categorical_encoder="ordinal",
        continuous_encoder: str = None,
        **kwargs,
    ) -> None:
        """
        Args:
            root (str): Path to the root of datasets for saving/loading.
            download (bool): If True, you must implement `self.download` method
                in the child class. Defaults to False.
        """
        self.seed = seed
        self.categorical_encoder = categorical_encoder
        self.continuous_encoder = continuous_encoder

        self.train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test.csv"))
        self.id = self.test["PassengerId"]

        self.train = self.train[self.columns + [self.target_column]]
        self.test = self.test[self.columns]

        self.label_encoder = LabelEncoder().fit(self.train[self.target_column])
        self.train[self.target_column] = self.label_encoder.transform(self.train[self.target_column])

    # def _init_checker(self):
    #     variables = ["continuous_columns", "categorical_columns", "binary_columns", "target_column", "data"]
    #     for variable in variables:
    #         if not hasattr(self, variable):
    #             if variable == "data":
    #                 if not (hasattr(self, "train") and hasattr(self, "test")):
    #                     raise ValueError("TabularDataFrame does not define `data`, but neither does `train`, `test`.")
    #             else:
    #                 raise ValueError(f"TabularDataFrame does not define a attribute: `{variable}`")

    # def show_data_details(self, train: pd.DataFrame, test: pd.DataFrame):
    #     all_data = pd.concat([train, test])
    #     logger.info(f"Dataset size       : {len(all_data)}")
    #     logger.info(f"All columns        : {all_data.shape[1] - 1}")
    #     logger.info(f"Num of cate columns: {len(self.categorical_columns)}")
    #     logger.info(f"Num of cont columns: {len(self.continuous_columns)}")

    #     y = all_data[self.target_column]
    #     class_ratios = y.value_counts(normalize=True)
    #     for label, class_ratio in zip(class_ratios.index, class_ratios.values):
    #         logger.info(f"class {label:<13}: {class_ratio:.3f}")

    def get_classify_dataframe(self) -> Dict[str, pd.DataFrame]:
        train = self.train
        test = self.test
        self.data_cate = pd.concat([train[self.categorical_columns], test[self.categorical_columns]])

        # self.show_data_details(train, test)
        classify_dfs = {
            "train": train,
            "test": test,
        }
        return classify_dfs

    def fit_feature_encoder(self, df_train):
        # Categorical values are fitted on all data.
        if self.categorical_columns != []:
            if self.categorical_encoder == "ordinal":
                self._categorical_encoder = OrdinalEncoder(dtype=np.int32).fit(self.data_cate)
            elif self.categorical_encoder == "onehot":
                self._categorical_encoder = OneHotEncoder(
                    sparse_output=False,
                    feature_name_combiner=feature_name_combiner,
                    dtype=np.int32,
                ).fit(self.data_cate)
            else:
                raise ValueError(self.categorical_encoder)
        if self.continuous_columns != [] and self.continuous_encoder is not None:
            if self.continuous_encoder == "standard":
                self._continuous_encoder = StandardScaler()
            elif self.continuous_encoder == "minmax":
                self._continuous_encoder = MinMaxScaler()
            else:
                raise ValueError(self.continuous_encoder)
            self._continuous_encoder.fit(df_train[self.continuous_columns])

    def apply_onehot_encoding(self, df: pd.DataFrame):
        encoded = self._categorical_encoder.transform(df[self.categorical_columns])
        encoded_columns = self._categorical_encoder.get_feature_names_out(self.categorical_columns)
        encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)
        df = df.drop(self.categorical_columns, axis=1)
        return pd.concat([df, encoded_df], axis=1)

    def apply_feature_encoding(self, dfs):
        for key in dfs.keys():
            if self.categorical_columns != []:
                if isinstance(self._categorical_encoder, OrdinalEncoder):
                    dfs[key][self.categorical_columns] = self._categorical_encoder.transform(
                        dfs[key][self.categorical_columns]
                    )
                else:
                    dfs[key] = self.apply_onehot_encoding(dfs[key])
            if self.continuous_columns != []:
                if self.continuous_encoder is not None:
                    dfs[key][self.continuous_columns] = self._continuous_encoder.transform(
                        dfs[key][self.continuous_columns]
                    )
                else:
                    dfs[key][self.continuous_columns] = dfs[key][self.continuous_columns].astype(np.float64)
        if self.categorical_columns != []:
            if isinstance(self._categorical_encoder, OneHotEncoder):
                self.categorical_columns = self._categorical_encoder.get_feature_names_out(self.categorical_columns)
        return dfs

    def processed_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Returns:
            dict[str, DataFrame]: The value has the keys "train", "val" and "test".
        """
        self.make_columns()
        self.fillnan()
        # self._init_checker()
        dfs = self.get_classify_dataframe()
        # preprocessing
        

        self.fit_feature_encoder(dfs["train"])
        dfs = self.apply_feature_encoding(dfs)
        self.all_columns = list(self.categorical_columns) + list(self.continuous_columns) + list(self.binary_columns)
        return dfs

    def get_categories_dict(self):
        if not hasattr(self, "_categorical_encoder"):
            return None

        categories_dict: Dict[str, List[Any]] = {}
        for categorical_column, categories in zip(self.categorical_columns, self._categorical_encoder.categories_):
            categories_dict[categorical_column] = categories.tolist()

        return categories_dict

    def fillnan(self) -> None:
        df_concat = pd.concat([self.train, self.test])

        # 欠損値処理オブジェクトを作成
        imputer_mode = SimpleImputer(strategy='most_frequent')
        imputer_mean = SimpleImputer(strategy='mean')

        if 'HomePlanet' in df_concat:
            df_concat['HomePlanet'] = imputer_mode.fit_transform(df_concat[['HomePlanet']])[:, 0]
        
        if 'CryoSleep' in df_concat:
            df_concat['CryoSleep'] = imputer_mode.fit_transform(df_concat[['CryoSleep']])[:, 0]

        if 'Cabin' in df_concat:
            ...
        
        if 'Destination' in df_concat:
            df_concat['Destination'] = imputer_mode.fit_transform(df_concat[['Destination']])[:, 0]   
        
        if 'Age' in df_concat:
            df_concat['Age'] = imputer_mean.fit_transform(df_concat[['Age']])[:, 0]
        
        if 'VIP' in df_concat:
            df_concat['VIP'] = imputer_mode.fit_transform(df_concat[['VIP']])[:, 0]
        
        if 'RoomService' in df_concat:
            df_concat['RoomService'] = imputer_mean.fit_transform(df_concat[['RoomService']])[:, 0]

        if 'FoodCourt' in df_concat:
            df_concat['FoodCourt'] = imputer_mean.fit_transform(df_concat[['FoodCourt']])[:, 0]
        
        if 'ShoppingMall' in df_concat:
            df_concat['ShoppingMall'] = imputer_mean.fit_transform(df_concat[['ShoppingMall']])[:, 0]

        if 'Spa' in df_concat:
            df_concat['Spa'] = imputer_mean.fit_transform(df_concat[['Spa']])[:, 0]
        
        if 'VRDeck' in df_concat:
            df_concat['VRDeck'] = imputer_mean.fit_transform(df_concat[['VRDeck']])[:, 0]    
        
        if 'Name' in df_concat:
            ...
        
        if 'Cabin_deck' in df_concat:
            df_concat['Cabin_deck'] = imputer_mode.fit_transform(df_concat[['Cabin_deck']])[:, 0]

        if 'Cabin_side' in df_concat:
            df_concat['Cabin_side'] = imputer_mode.fit_transform(df_concat[['Cabin_side']])[:, 0]
        
        self.train = df_concat[:len(self.train)]
        self.test = df_concat[len(self.train):]
        self.test.drop(self.target_column, axis=1, inplace=True)
    
    def make_columns(self) -> None:
        ...

        

class V0(TabularDataFrame):
    continuous_columns = [
        'Age',
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck'
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class V1(TabularDataFrame):
    continuous_columns = [
        'Age',
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck'
    ]

    categorical_columns = [
        'HomePlanet',
        'CryoSleep',
        'Cabin',
        'Destination',
        'VIP',
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def make_columns(self) -> None:
        df_concat = pd.concat([self.train, self.test])
        df_concat['Cabin_deck'] = df_concat['Cabin'].str[0]
        df_concat['Cabin_side'] = df_concat['Cabin'].str[-1]

        df_concat.drop('Cabin', axis=1, inplace=True)
        self.categorical_columns = [col for col in self.categorical_columns if col !='Cabin']
        self.categorical_columns.extend(['Cabin_deck', 'Cabin_side'])


        self.train = df_concat[:len(self.train)]
        self.test = df_concat[len(self.train):]
        self.test.drop(self.target_column, axis=1, inplace=True)
    
    



    
