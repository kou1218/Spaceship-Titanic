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
                    handle_unknown='error', 
                    drop='first',
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
        
        self.add_new_feature()

        # self.make_binary_columns()
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

    def make_binary_columns(self) -> None:
        ...

    def add_new_feature(self) -> None:
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


class V2(TabularDataFrame):
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
        'Cabin',
        'Destination',
        'VIP',
        'CryoSleep'
    ]

    binary_columns = [
        # 'VIP',
        # 'CryoSleep'
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def fillnan(self) -> None:
        df_concat = pd.concat([self.train, self.test])

        # 欠損値処理オブジェクトを作成
        imputer_mode = SimpleImputer(strategy='most_frequent')
        imputer_mean = SimpleImputer(strategy='mean')

        if 'RoomService' in df_concat:
            # df_concat['RoomService'] = imputer_mean.fit_transform(df_concat[['RoomService']])[:, 0]
            # df_concat.loc[df_concat['CryoSleep'] == True, ['RoomService']] = 0
            df_concat['RoomService'].fillna(0, inplace=True)
        
        if 'FoodCourt' in df_concat:
            # df_concat['FoodCourt'] = imputer_mean.fit_transform(df_concat[['FoodCourt']])[:, 0]
            # df_concat.loc[df_concat['CryoSleep'] == True, ['FoodCourt']] = 0
            df_concat['FoodCourt'].fillna(0, inplace=True)
        
        if 'ShoppingMall' in df_concat:
            # df_concat['ShoppingMall'] = imputer_mean.fit_transform(df_concat[['ShoppingMall']])[:, 0]
            # df_concat.loc[df_concat['CryoSleep'] == True, ['ShoppingMall']] = 0
            df_concat['ShoppingMall'].fillna(0, inplace=True)

        if 'Spa' in df_concat:
            # df_concat['Spa'] = imputer_mean.fit_transform(df_concat[['Spa']])[:, 0]
            # df_concat.loc[df_concat['CryoSleep'] == True, ['Spa']] = 0
            df_concat['Spa'].fillna(0, inplace=True)
        
        if 'VRDeck' in df_concat:
            # df_concat['VRDeck'] = imputer_mean.fit_transform(df_concat[['VRDeck']])[:, 0]
            # df_concat.loc[df_concat['CryoSleep'] == True, ['VRDeck']] = 0
            df_concat['VRDeck'].fillna(0, inplace=True)
        
        if 'HomePlanet' in df_concat:
            # df_concat['HomePlanet'] = imputer_mode.fit_transform(df_concat[['HomePlanet']])[:, 0]

            # vs_code上では上がったが提出したら下がった
            # # Cabin_deckがA,B,Cの場合はEuropaで埋める
            # df_concat.loc[df_concat['Cabin_deck'].isin(['A', 'B', 'C']), 'HomePlanet'] = 'Europa'
            # # それ以外の場合はEarthで埋める
            # df_concat['HomePlanet'].fillna('Earth', inplace=True)

            # 今のところ一番提出スコアが良い
            # HomePlanet の欠損値を処理
            for i, row in df_concat.iterrows():
                if pd.isnull(row['HomePlanet']):
                    if row['Cabin_deck'] in ['A', 'B', 'C']:
                        df_concat.at[i, 'HomePlanet'] = 'Europa'
                    elif row['Cabin_deck'] in ['D']:
                        # Mars か Europa をランダムに選ぶ
                        df_concat.at[i, 'HomePlanet'] = np.random.choice(['Mars', 'Europa'], p=[0.66, 0.34])
                    elif row['Cabin_deck'] in ['E']:
                        # ランダムに選ぶ
                        df_concat.at[i, 'HomePlanet'] = np.random.choice(['Mars', 'Europa', 'Earth'], p=[0.40, 0.15, 0.45])
                    elif row['Cabin_deck'] in ['F']:
                        # Mars か Earth をランダムに選ぶ
                        df_concat.at[i, 'HomePlanet'] = np.random.choice(['Mars', 'Earth'], p=[0.4, 0.6])
                    elif row['Cabin_deck'] in ['G']:
                        df_concat.at[i, 'HomePlanet'] = 'Earth'
                    elif row['Cabin_deck'] in ['T']:
                        df_concat.at[i, 'HomePlanet'] = 'Europa'

            # vscode上では下がる
            df_concat['HomePlanet'] = imputer_mode.fit_transform(df_concat[['HomePlanet']])[:, 0]   
        
        if 'CryoSleep' in df_concat:
            df_concat.loc[(df_concat['RoomService'] == 0) & (df_concat['FoodCourt'] == 0) & (df_concat['ShoppingMall'] == 0) & (df_concat['Spa'] == 0) & (df_concat['VRDeck'] == 0), ['CryoSleep']] = True
            # df_concat['CryoSleep'] = imputer_mode.fit_transform(df_concat[['CryoSleep']])[:, 0]
            df_concat['CryoSleep'].fillna(False, inplace=True)


        if 'Cabin' in df_concat:
            ...
        
        if 'Destination' in df_concat:
            # df_concat['Destination'] = imputer_mode.fit_transform(df_concat[['Destination']])[:, 0] 

            # vscode上では上がるが提出したら下がる
            # Destination の欠損値を処理
            for i, row in df_concat.iterrows():
                if pd.isnull(row['Destination']):
                    if row['HomePlanet'] in ['Europa']:
                        df_concat.at[i, 'Destination'] = np.random.choice(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], p=[0.574, 0.009, 0.417])
                    elif row['HomePlanet'] in ['Mars']:
                        df_concat.at[i, 'Destination'] = np.random.choice(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], p=[0.862, 0.026, 0.112])
                    elif row['HomePlanet'] in ['Earth']:
                        df_concat.at[i, 'Destination'] = np.random.choice(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], p=[0.7, 0.15, 0.15])  
        
        if 'Age' in df_concat:
            df_concat['Age'] = imputer_mean.fit_transform(df_concat[['Age']])[:, 0]
        
        if 'VIP' in df_concat:
            df_concat['VIP'] = imputer_mode.fit_transform(df_concat[['VIP']])[:, 0]
          
        
        if 'Name' in df_concat:
            ...

        if 'Cabin_deck' in df_concat:
            # Cabin_deck の欠損値を処理
            for i, row in df_concat.iterrows():
                if pd.isnull(row['Cabin_deck']):
                    if row['HomePlanet'] in ['Europa']:
                        df_concat.at[i, 'Cabin_deck'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], p=[0.12, 0.37, 0.37, 0.08, 0.06])
                    elif row['HomePlanet'] in ['Mars']:
                        df_concat.at[i, 'Cabin_deck'] = np.random.choice(['E', 'F', 'D'], p=[0.2, 0.2, 0.6])
                    elif row['HomePlanet'] in ['Earth']:
                        df_concat.at[i, 'Cabin_deck'] = np.random.choice(['E', 'F', 'G'], p=[0.1, 0.35, 0.55])

            df_concat['Cabin_deck'] = imputer_mode.fit_transform(df_concat[['Cabin_deck']])[:, 0]        

        if 'Cabin_side' in df_concat:
            # df_concat['Cabin_side'] = imputer_mode.fit_transform(df_concat[['Cabin_side']])[:, 0]

            # Cabin_sideカラムの欠損値を含む行を特定
            missing_rows = df_concat['Cabin_side'].isnull()

            # 欠損値を'P'または'S'のいずれかでランダムに埋める
            random_values = np.random.choice(['P', 'S'], size=len(df_concat))
            df_concat.loc[missing_rows, 'Cabin_side'] = random_values[missing_rows]

        if 'Cabin_num' in df_concat:
            # df_concat['Cabin_num'].fillna(10000,inplace=True)

            cabin_deck_dict = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'T': 0}

            for i, row in df_concat.iterrows():
                letter = df_concat['Cabin_deck'].iloc[i]
                if letter in cabin_deck_dict:
                    cabin_deck_dict[letter] += 1

                if pd.isnull(row['Cabin_num']):
                    df_concat.at[i,'Cabin_num'] = cabin_deck_dict[letter]

            
            df_concat['Cabin_num'] = df_concat['Cabin_num'].astype(int)


        self.train = df_concat[:len(self.train)]
        self.test = df_concat[len(self.train):]
        self.test.drop(self.target_column, axis=1, inplace=True)

        # print(df_concat['PassengerId'].head())
        # print(df_concat['Family'].head(4276))
        # print(df_concat.isnull().sum())
        # exit()       

        
    
    def make_columns(self) -> None:
        df_concat = pd.concat([self.train, self.test])
        df_concat[["Cabin_deck", "Cabin_num", "Cabin_side"]] = df_concat["Cabin"].str.split("/", expand=True)

        df_concat.drop('Cabin', axis=1, inplace=True)
        self.categorical_columns = [col for col in self.categorical_columns if col !='Cabin']
        self.categorical_columns.extend(['Cabin_deck', 'Cabin_side'])
        self.continuous_columns.extend(['Cabin_num'])

        # ファミリーIDと家族の人数を追加
        df_concat['Family'] = df_concat['PassengerId'].apply(lambda x: x.split('_')[0])
        family_counts = df_concat['Family'].value_counts()
        df_concat['FamilySize'] = df_concat['Family'].apply(lambda x: family_counts[x])
        df_concat['PassengerId'] = df_concat['PassengerId'].astype(int)
        # self.continuous_columns.extend(['Family'])
        self.categorical_columns.extend(['FamilySize'])

        

        # df_concat.drop('VIP', axis=1, inplace=True)
        # self.categorical_columns = [col for col in self.categorical_columns if col !='VIP']
        # df_concat['VIP'] = df_concat['VIP'].map({'True': True, 'False': False})
        # df_concat['CryoSleep'] = df_concat['CryoSleep'].map({'True': True, 'False': False})



        self.train = df_concat[:len(self.train)]
        self.test = df_concat[len(self.train):]
        self.test.drop(self.target_column, axis=1, inplace=True)

    
    def make_binary_columns(self) -> None:
        df_concat = pd.concat([self.train, self.test])

        df_concat['VIP'] = df_concat['VIP'].astype('bool')

        df_concat['CryoSleep'] = df_concat['CryoSleep'].astype('bool')

        self.train = df_concat[:len(self.train)]
        self.test = df_concat[len(self.train):]
        self.test.drop(self.target_column, axis=1, inplace=True)

    
    def add_new_feature(self) -> None:
        df_concat = pd.concat([self.train, self.test])

        # add特徴量
        df_concat['HomePlanet_CryoSleep'] = df_concat['HomePlanet'] * df_concat['CryoSleep']
        self.categorical_columns.extend(['HomePlanet_CryoSleep'])

        # add特徴量
        df_concat['TotalSpent'] = df_concat['RoomService'] + df_concat['FoodCourt'] + df_concat['ShoppingMall'] + df_concat['Spa'] + df_concat['VRDeck']
        self.continuous_columns.extend(['TotalSpent'])
        # # vscode上で精度ギャン下がり
        # df_concat.drop('RoomService', axis=1, inplace=True)
        # self.continuous_columns = [col for col in self.continuous_columns if col !='RoomService']
        # df_concat.drop('FoodCourt', axis=1, inplace=True)
        # self.continuous_columns = [col for col in self.continuous_columns if col !='FoodCourt']
        # df_concat.drop('ShoppingMall', axis=1, inplace=True)
        # self.continuous_columns = [col for col in self.continuous_columns if col !='ShoppingMall']
        # df_concat.drop('Spa', axis=1, inplace=True)
        # self.continuous_columns = [col for col in self.continuous_columns if col !='Spa']
        # df_concat.drop('VRDeck', axis=1, inplace=True)
        # self.continuous_columns = [col for col in self.continuous_columns if col !='VRDeck']

        # add特徴量
        df_concat['Cabin_deck_CryoSleep'] = df_concat['Cabin_deck'] * df_concat['CryoSleep']
        self.categorical_columns.extend(['Cabin_deck_CryoSleep'])

        # add特徴量
        df_concat['HomePlanet_Cabin_deck'] = df_concat['HomePlanet'] + df_concat['Cabin_deck']
        self.categorical_columns.extend(['HomePlanet_Cabin_deck'])

        # add特徴量
        df_concat['Cabin_deck_side'] = df_concat['Cabin_deck'] + df_concat['Cabin_side']
        self.categorical_columns.extend(['Cabin_deck_side'])

        bins = [0, 16, 28, 36, 46, 58, 66, df_concat['Age'].max()+1]
        labels = ['0-16', '17-28', '29-36', '37-46', '47-58', '59-66', '67+']
        df_concat['Age_bin'] = pd.cut(df_concat['Age'], bins=bins, labels=labels, right=False)
        self.categorical_columns.extend(['Age_bin'])
        df_concat.drop('Age', axis=1, inplace=True)
        self.continuous_columns = [col for col in self.continuous_columns if col !='Age']

        bins = [0, 320, 640, 800, 1180, 1500, 1820, 2000 ,df_concat['Cabin_num'].max()+1]
        labels = ['0-320', '320-640', '640-800', '800-1180', '1180-1500', '1500-1820', '1800-2000', '2000+']
        df_concat['Cabin_num_bin'] = pd.cut(df_concat['Cabin_num'], bins=bins, labels=labels, right=False)
        self.categorical_columns.extend(['Cabin_num_bin'])
        df_concat.drop('Cabin_num', axis=1, inplace=True)
        self.continuous_columns = [col for col in self.continuous_columns if col !='Cabin_num']

        # bins = [0, 500, 1000, 1500, 2000, df_concat['Cabin_num'].max()+1]
        # labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000+']
        # df_concat['Cabin_num_bin'] = pd.cut(df_concat['Cabin_num'], bins=bins, labels=labels, right=False)
        # self.categorical_columns.extend(['Cabin_num_bin'])

        # add特徴量
        df_concat['Destination_CryoSleep'] = df_concat['Destination'] * df_concat['CryoSleep']
        self.categorical_columns.extend(['Destination_CryoSleep'])


        # cvは上がるが提出時下がる
        # # add特徴量
        # df_concat['HomePlanet_Destination'] = df_concat['HomePlanet'] + df_concat['Destination']
        # self.categorical_columns.extend(['HomePlanet_Destination'])

        # add特徴量
        # df_concat['Cabin_deck_Destination'] = df_concat['Cabin_deck'] + df_concat['Destination']
        # self.categorical_columns.extend(['Cabin_deck_Destination'])



        # print(df_concat.head())
        # print(df_concat[df_concat['Age_bin'].isnull()])
        # print(df_concat.isnull().sum())
        # exit()
        
        self.train = df_concat[:len(self.train)]
        self.test = df_concat[len(self.train):]
        self.test.drop(self.target_column, axis=1, inplace=True)

        
    



    
