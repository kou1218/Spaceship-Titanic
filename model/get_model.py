from experiment.utils import set_seed


from .model import XGBoostClassifier, LightGBMClassifier, CBTClassifier, RFClassifier
from .model import GBClassifier, ETClassifier, KNClassifier



def get_classifier(name, *, input_dim, output_dim, model_config, seed=42, verbose=0):
    set_seed(seed=seed)
    if name == "xgboost":
        return XGBoostClassifier(input_dim, output_dim, model_config, verbose)
    elif name == "lightgbm":
        return LightGBMClassifier(input_dim, output_dim, model_config, verbose)
    elif name == "catboost":
        return CBTClassifier(input_dim, output_dim, model_config, verbose)
    elif name == "randomforest":
        return RFClassifier(input_dim, output_dim, model_config, verbose)
    elif name == "gradientboosting":
        return GBClassifier(input_dim, output_dim, model_config, verbose)
    elif name == "extratrees":
        return ETClassifier(input_dim, output_dim, model_config, verbose)
    elif name == "kneighbors":
        return KNClassifier(input_dim, output_dim, model_config, verbose)
    elif name == "xgboost2":
        return XGBoostClassifier(input_dim, output_dim, model_config, verbose)
    elif name == "catboost_optuna":
        return CBTClassifier(input_dim, output_dim, model_config, verbose)
    # elif name == "xgblgbm":
    #     return XGBLGBMClassifier(input_dim, output_dim, model_config, verbose)
    # elif name == "xgb5lgbm5":
    #     return XGB5LGBM5Classifier(input_dim, output_dim, model_config, verbose)
    # elif name == "lgbm10":
    #     return LGBM10Classfier(input_dim, output_dim, model_config, verbose)
    else:
        raise KeyError(f"{name} is not defined.")