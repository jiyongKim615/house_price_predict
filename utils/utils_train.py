import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import RepeatedKFold
from pyod.models.ecod import ECOD
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.integration import XGBoostPruningCallback
import lightgbm as lgb

# Validation function
from utils.utils_optuna import XGBRegressorOptuna, CatBoostRegressorOptuna

n_folds = 5

FS = (14, 6)  # figure size
RS = 124  # random state
N_JOBS = 8  # number of parallel threads

# repeated K-folds
N_SPLITS = 10
N_REPEATS = 1

# Optuna
N_TRIALS = 100
MULTIVARIATE = True

# XGBoost
EARLY_STOPPING_ROUNDS = 100


def rmsle_cv(model, train_df, y_train_df):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.values)
    rmse = np.sqrt(-cross_val_score(model, train_df.values, y_train_df, scoring="neg_mean_squared_error", cv=kf))
    return rmse


def get_robust_pipeline():
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

    return lasso, ENet


def get_outlier_label_by_ecod(train_df, test_df, y_train_df):
    clf = ECOD()
    clf.fit(train_df)
    pred = clf.predict(train_df)
    train_df['OUTLIER'] = pred
    no_outlier_train_df = train_df[train_df['OUTLIER'] == 0]
    no_outlier_index = no_outlier_train_df.index.tolist()
    no_outlier_y_train_df = y_train_df[no_outlier_index]
    clf2 = ECOD()
    clf2.fit(test_df)
    pred2 = clf2.predict(test_df)
    test_df['OUTLIER'] = pred2
    return train_df, no_outlier_train_df, no_outlier_y_train_df, test_df


# 스태킹 & OOF 에 대한 평가 함수 정의
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# 메타 모델 구축 & 스태킹 & OOF
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


def final_xgboost(X_train, y_train, X_test, category_feature):
    xgboptuna_reg = XGBRegressorOptuna()

    params, preds_xgb = xgboptuna_reg.optimize(X_train,
                                               y_train,
                                               test_data=X_test,
                                               # int, str 타입 이어야 한다. float는 허용하지 않음
                                               cat_features=category_feature,
                                               eval_metric='rmse', n_trials=3)
    model_xgb = xgb.XGBRegressor(**params)
    return model_xgb


def final_catboost(X_train, y_train, X_test, category_feature):
    catboostoptuna_reg = CatBoostRegressorOptuna(use_gpu=False)

    params, preds_cat = catboostoptuna_reg.optimize(X_train,
                                                    y_train,
                                                    test_data=X_test,
                                                    # int, str 타입 이어야 한다. float는 허용하지 않음
                                                    cat_features=category_feature,
                                                    eval_metric='rmse', n_trials=3)

    model_cat = CatBoostRegressor()
    model_cat.set_params(**params)
    return model_cat