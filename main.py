from train.preprocess import get_final_train_test
from uncertainty_estimation.ensemble_model_utils import get_uncertainty_interval
from uncertainty_estimation.lgb_distributions.Expectile import Expectile
from uncertainty_estimation.uncertatinty_main import plot_train_df_with_y_test
from uncertainty_estimation.utils_uncertainty import get_all_predictions, plot_result
from utils.utils_train import get_robust_pipeline

if __name__ == '__main__':
    X_train, X_test, y_train = get_final_train_test()
    plot_train_df_with_y_test