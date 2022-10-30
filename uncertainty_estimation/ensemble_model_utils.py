from uncertainty_estimation.xgboost_distributions.Expectile import Expectile
from uncertainty_estimation.lgb_distributions.Expectile import Expectile2
from uncertainty_estimation.xgb_model import *
from uncertainty_estimation.lgb_model import *
import multiprocessing


def get_mean_interval_info(pred_expectile_xgb, pred_expectile_lgb):
    mean_y_lower = (pred_expectile_xgb['expectile_0.05'] + pred_expectile_lgb['expectile_0.05']) / 2
    mean_y_upper = (pred_expectile_xgb['expectile_0.95'] + pred_expectile_lgb['expectile_0.95']) / 2
    return mean_y_lower, mean_y_upper


def get_uncertainty_interval(X_train_df_new, y_train_df, X_test_df_new, xgb_params, lgb_params):
    np.random.seed(123)
    distribution = Expectile
    distribution.expectiles = [0.05, 0.95]  # Expectiles to be estimated: needs to be a list of at least two expectiles.
    distribution.stabilize = "None"  # Option to stabilize Gradient/Hessian. Options are "None", "MAD", "L2".

    distribution2 = Expectile2
    distribution2.expectiles = [0.05,
                                0.95]  # Expectiles to be estimated: needs to be a list of at least two expectiles.
    distribution2.stabilize = "None"  # Option to stabilize Gradient/Hessian. Options are "None", "MAD", "L2".

    n_cpu = multiprocessing.cpu_count()
    dtrain = xgb.DMatrix(X_train_df_new, label=y_train_df, nthread=n_cpu)
    dtest = xgb.DMatrix(X_test_df_new, nthread=n_cpu)

    dtrain_lgbm = lgb.Dataset(X_train_df_new, label=y_train_df)

    # Train Model with optimized hyper-parameters
    xgboostlss_model = xgboostlss.train(xgb_params,
                                        dtrain,
                                        dist=distribution,
                                        num_boost_round=100)

    # Extract predicted expectiles
    pred_expectile_xgb = xgboostlss.predict(xgboostlss_model,
                                            dtest,
                                            dist=distribution,
                                            pred_type="expectiles")

    # Train Model with optimized hyper-parameters
    lightgbmlss_model = lightgbmlss.train(lgb_params,
                                          dtrain_lgbm,
                                          dist=distribution2,
                                          num_boost_round=100)

    # Extract predicted expectiles
    pred_expectile_lgb = lightgbmlss.predict(lightgbmlss_model,
                                             dtest=X_test_df_new,
                                             dist=distribution2,
                                             pred_type="expectiles")

    return pred_expectile_xgb, pred_expectile_lgb
