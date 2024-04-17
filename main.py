import logging

import hydra

import experiment
from experiment import ExpBase

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="main", version_base="1.1")
def main(config):
    exp: ExpBase = getattr(experiment, config.exp.name)(config)
    exp.run()


if __name__ == "__main__":
    main()

# def main():
#     dataframe = v1()
#     dataframe.make_columns()
  

#     train = dataframe.train
#     test = dataframe.test
#     feature_columns = dataframe.feature_columns
#     target_column = dataframe.target_column

#     model = XGBoostClassifier(input_dim=6, output_dim=2, verbose=1)

#     df_train, df_val = train_test_split(train, test_size=0.2)

#     model.fit(df_train[feature_columns], df_train[target_column], eval_set=[df_val[feature_columns], df_val[target_column]])


    


































