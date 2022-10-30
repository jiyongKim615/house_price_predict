from uncertainty_estimation.ensemble_model_utils import get_uncertainty_interval, get_mean_interval_info
from uncertainty_estimation.utils_uncertainty import plot_result
from utils.utils_train import tune_all_model
from sklearn.model_selection import train_test_split


def plot_train_df_with_y_test(X_train_df_new, y_train_df, category_feature):
    X_train, X_test, y_train, y_test = train_test_split(X_train_df_new, y_train_df, test_size=0.3, random_state=42)
    model_xgb, xgb_params, model_lgbm, lgb_params = \
        tune_all_model(X_train, y_train, X_test, category_feature)

    y_pred = 0.5 * model_xgb.predict(X_test) + 0.5 * model_lgbm.predict(X_test)
    pred_expectile_xgb, pred_expectile_lgb = \
        get_uncertainty_interval(X_train, y_train,
                                 X_test, xgb_params, lgb_params)
    mean_y_upper, mean_y_lower = get_mean_interval_info(pred_expectile_xgb, pred_expectile_lgb)
    plot_result(X_train, y_train, X_test,
                y_test, y_pred, mean_y_upper, mean_y_lower)


def get_x_test_interval_only_by_xgb_lgbm(X_train_df_new, y_train_df, X_test_df_new, category_feature):
    model_xgb, xgb_params, model_lgbm, lgb_params = \
        tune_all_model(X_train_df_new, y_train_df, X_test_df_new, category_feature)

    pred_expectile_xgb, pred_expectile_lgb = \
        get_uncertainty_interval(X_train_df_new, y_train_df,
                                 X_test_df_new, xgb_params, lgb_params)

    mean_y_lower, mean_y_upper = \
        get_mean_interval_info(pred_expectile_xgb, pred_expectile_lgb)

    return mean_y_lower, mean_y_upper
