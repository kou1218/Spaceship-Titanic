import data.data as data
from data import TableDataFrame, v1

from model import XGBoostClassifier

from sklearn.model_selection import train_test_split

class ExpBase:
    def run():
        dataframe = v1()
        dataframe.make_columns()
    

        train = dataframe.train
        test = dataframe.test
        feature_columns = dataframe.feature_columns
        target_column = dataframe.target_column

        model = XGBoostClassifier(input_dim=6, output_dim=2, verbose=1)

        df_train, df_val = train_test_split(train, test_size=0.2)

        model.fit(df_train[feature_columns], df_train[target_column], eval_set=[df_val[feature_columns], df_val[target_column]])