import pandas as pd 
from sklearn.preprocessing import LabelEncoder

import datasets

class TableDataFrame():
    def __init__(self):
        self.train = pd.read_csv('datasets/train.csv')
        self.test = pd.read_csv('datasets/test.csv')
        self.target_column = 'Transported'
        self.feature_columns = [col for col in self.train.columns if col != self.target_column]

        self.le = LabelEncoder()

class v1(TableDataFrame):
    def __init__(self):
        super(v1, self).__init__()
    
    def make_columns(self):
        self.feature_columns = [
            'Age',
            'RoomService',
            'FoodCourt',
            'ShoppingMall',
            'Spa',
            'VRDeck',
            ]
        
        self.train['Transported'] = self.le.fit_transform(self.train['Transported'].values)
        self.train = pd.concat([self.train[self.feature_columns], self.train['Transported']], axis=1)

    


    
