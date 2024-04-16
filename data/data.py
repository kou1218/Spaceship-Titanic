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

from .utils import feature_name_combiner

logger = logging.getLogger(__name__)


# Copied from https://github.com/pfnet-research/deep-table.
# Modified by somaonishi and shoyameguro.
class TabularDataFrame(object):
    columns = [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "time_signature",
        "song_name",
    ]
    continuous_columns = []
    categorical_columns = []
    binary_columns = []
    target_column = "genre"

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
        self.id = self.test["ID"]

        self.train = self.train[self.columns + [self.target_column]]
        self.test = self.test[self.columns]

        self.label_encoder = LabelEncoder().fit(self.train[self.target_column])
        self.train[self.target_column] = self.label_encoder.transform(self.train[self.target_column])

    def _init_checker(self):
        variables = ["continuous_columns", "categorical_columns", "binary_columns", "target_column", "data"]
        for variable in variables:
            if not hasattr(self, variable):
                if variable == "data":
                    if not (hasattr(self, "train") and hasattr(self, "test")):
                        raise ValueError("TabularDataFrame does not define `data`, but neither does `train`, `test`.")
                else:
                    raise ValueError(f"TabularDataFrame does not define a attribute: `{variable}`")

    def show_data_details(self, train: pd.DataFrame, test: pd.DataFrame):
        all_data = pd.concat([train, test])
        logger.info(f"Dataset size       : {len(all_data)}")
        logger.info(f"All columns        : {all_data.shape[1] - 1}")
        logger.info(f"Num of cate columns: {len(self.categorical_columns)}")
        logger.info(f"Num of cont columns: {len(self.continuous_columns)}")

        y = all_data[self.target_column]
        class_ratios = y.value_counts(normalize=True)
        for label, class_ratio in zip(class_ratios.index, class_ratios.values):
            logger.info(f"class {label:<13}: {class_ratio:.3f}")

    def get_classify_dataframe(self) -> Dict[str, pd.DataFrame]:
        train = self.train
        test = self.test
        self.data_cate = pd.concat([train[self.categorical_columns], test[self.categorical_columns]])

        self.show_data_details(train, test)
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
        self._init_checker()
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


class V0(TabularDataFrame):
    continuous_columns = [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "time_signature",
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class V1(TabularDataFrame):
    def set_continuous_columns(self):
        continuous_columns = [
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "duration_ms",
            "time_signature",
        ]
        # The title of this song is "{song}".
        song_embed = np.load(to_absolute_path("datasets/song_embed.npy"))
        self.song_embed = pd.DataFrame(song_embed, columns=[f"song_embed_{i}" for i in range(song_embed.shape[1])])
        self.continuous_columns = continuous_columns + self.song_embed.columns.tolist()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.set_continuous_columns()
        self.train = pd.concat([self.train, self.song_embed.iloc[: len(self.train)]], axis=1)
        self.test = pd.concat([self.test, self.song_embed.iloc[len(self.train) :].reset_index(drop=True)], axis=1)


class V2(TabularDataFrame):
    def set_continuous_columns(self):
        continuous_columns = [
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "duration_ms",
            "time_signature",
        ]
        # Solve the task of estimating the genre of this song.
        # The genres given are Dark Trap, Emo, Hiphop, Pop, Rap, RnB, Trap Metal, and Underground Rap.
        # The title of this song is "{song}".
        song_embed = np.load(to_absolute_path("datasets/song_paragraph_embed.npy"))
        self.song_embed = pd.DataFrame(song_embed, columns=[f"song_embed_{i}" for i in range(song_embed.shape[1])])
        self.continuous_columns = continuous_columns + self.song_embed.columns.tolist()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.set_continuous_columns()
        self.train = pd.concat([self.train, self.song_embed.iloc[: len(self.train)]], axis=1)
        self.test = pd.concat([self.test, self.song_embed.iloc[len(self.train) :].reset_index(drop=True)], axis=1)
    


    


    
