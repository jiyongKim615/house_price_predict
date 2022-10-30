import pandas as pd
import numpy as np

from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_consistent_length, column_or_1d, assert_all_finite
from sklearn.utils.extmath import stable_cumsum
import matplotlib.pyplot as plt
from utils.utils_train import final_catboost


def calc_uncertainty_regection_curve(errors, uncertainty, group_by_uncertainty=True):
    n_objects = errors.shape[0]
    if group_by_uncertainty:
        data = pd.DataFrame(dict(
            errors=errors,
            uncertainty=uncertainty
        ))
        mean_errors = data.groupby("uncertainty").mean()
        mean_errors.rename(columns={"errors": "mean_errors"}, inplace=True)
        data = data.join(mean_errors, "uncertainty")
        data.drop("errors", axis=1, inplace=True)

        uncertainty_order = data["uncertainty"].argsort()
        errors = data["mean_errors"][uncertainty_order]
    else:
        uncertainty_order = uncertainty.argsort()
        errors = errors[uncertainty_order]

    error_rates = np.zeros(n_objects + 1)
    error_rates[:-1] = np.cumsum(errors)[::-1] / n_objects
    return error_rates


assert np.allclose(
    calc_uncertainty_regection_curve(np.array([2, 1]), np.array([1, 0])).mean(),
    2 / 3
)
assert np.allclose(
    calc_uncertainty_regection_curve(np.arange(5), np.array([0, 0, 2, 1, 1])).mean(),
    0.8
)
debug_errors = np.random.rand(10)
assert np.allclose(
    calc_uncertainty_regection_curve(debug_errors, np.zeros_like(debug_errors)).mean(),
    debug_errors.mean() / 2
)


def calc_aucs(errors, uncertainty):
    uncertainty_rejection_curve = calc_uncertainty_regection_curve(errors, uncertainty)
    uncertainty_rejection_auc = uncertainty_rejection_curve.mean()
    random_rejection_auc = uncertainty_rejection_curve[0] / 2
    ideal_rejection_auc = calc_uncertainty_regection_curve(errors, errors).mean()

    rejection_ratio = (uncertainty_rejection_auc - random_rejection_auc) / (
            ideal_rejection_auc - random_rejection_auc) * 100.0
    return rejection_ratio, uncertainty_rejection_auc


def prr_classification(labels, probs, measure, rev: bool):
    if rev:
        measure = -measure
    preds = np.argmax(probs, axis=1)
    errors = (labels != preds).astype("float32")
    return calc_aucs(errors, measure)


def prr_regression(targets, preds, measure):
    preds = np.squeeze(preds)
    # Compute MSE errors
    errors = (preds - targets) ** 2
    return calc_aucs(errors, measure)


def ood_detect(domain_labels, in_measure, out_measure, mode='ROC', pos_label=1):
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores)
    if pos_label != 1:
        scores *= -1.0

    if mode == 'PR':
        precision, recall, thresholds = precision_recall_curve(domain_labels, scores)
        aupr = auc(recall, precision)
        return aupr

    elif mode == 'ROC':
        roc_auc = roc_auc_score(domain_labels, scores)
        return roc_auc


def nll_regression(target, mu, var, epsilon=1e-8, raw=False):
    nll = (target - mu) ** 2 / (2.0 * var + epsilon) + np.log(var + epsilon) / 2.0 + np.log(2 * np.pi) / 2.0
    if raw:
        return nll
    return np.mean(nll)


def nll_class(target, probs, epsilon=1e-10):
    log_p = -np.log(probs + epsilon)
    return target * log_p[:, 1] + (1 - target) * log_p[:, 0]


def ens_nll_regression(target, preds, epsilon=1e-8, raw=False):
    mu = preds[:, :, 0]
    var = preds[:, :, 1]
    nll = (target - mu) ** 2 / (2.0 * var + epsilon) + np.log(var + epsilon) / 2.0 + np.log(2 * np.pi) / 2.0
    proba = np.exp(-1 * nll)
    if raw:
        return -1 * np.log(np.mean(proba, axis=0))
    return np.mean(-1 * np.log(np.mean(proba, axis=0)))


def calc_rmse(preds, target, raw=False):
    if raw:
        return (preds - target) ** 2
    return np.sqrt(np.mean((preds - target) ** 2))


def ens_rmse(target, preds, epsilon=1e-8, raw=False):
    means = preds[:, :, 0]  # mean values predicted by all models
    avg_mean = np.mean(means, axis=0)  # average predicted mean value
    if raw:
        return calc_rmse(avg_mean, target, raw=True)
    return calc_rmse(avg_mean, target)


def _check_pos_label_consistency(pos_label, y_true):
    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in 'OUS' or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError(
            f"y_true takes value in {{{classes_repr}}} and pos_label is not "
            f"specified: either make y_true take value in {{0, 1}} or "
            f"{{-1, 1}} or pass pos_label explicitly."
        )
    elif pos_label is None:
        pos_label = 1.0

    return pos_label


def _binary_clf_curve_ret(y_true, y_score, pos_label=None, sample_weight=None):
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    pos_label = _check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    # distinct_value_indices = np.where(np.diff(y_score))[0]
    # threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)  # [threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)  # [threshold_idxs]
    else:
        fps = stable_cumsum((1 - y_true))  # [threshold_idxs]
    return fps, tps, y_score  # [threshold_idxs]


def _precision_recall_curve_retention(y_true, probas_pred, *, pos_label=None,
                                      sample_weight=None):
    fps, tps, thresholds = _binary_clf_curve_ret(y_true, probas_pred,
                                                 pos_label=pos_label,
                                                 sample_weight=sample_weight)

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(-1, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def _acceptable_error(errors, threshold):
    return np.asarray(errors <= threshold, dtype=np.float32)


def _calc_fbeta_regection_curve(errors, uncertainty, threshold, beta=1.0, group_by_uncertainty=True, eps=1e-10):
    ae = _acceptable_error(errors, threshold)
    pr, rec, _ = _precision_recall_curve_retention(ae, -uncertainty)
    pr = np.asarray(pr)
    rec = np.asarray(rec)
    f_scores = (1 + beta ** 2) * pr * rec / (pr * beta ** 2 + rec + eps)

    return f_scores, pr, rec


def f_beta_metrics(errors, uncertainty, threshold, beta=1.0):
    """

    :param errors: Per sample errors - array [n_samples]
    :param uncertainty: Uncertainties associated with each prediction. rray [n_samples]
    :param threshold: The error threshold below which we consider the prediction acceptable
    :param beta: The beta value for the F_beta metric. Defaults to 1
    :return: fbeta_auc, fbeta_95, retention
    """
    f_scores, pr, rec = _calc_fbeta_regection_curve(errors, uncertainty, threshold, beta)
    ret = np.arange(pr.shape[0]) / pr.shape[0]

    f_auc = auc(ret[::-1], f_scores)
    f95 = f_scores[::-1][np.int(0.95 * pr.shape[0])]

    return f_auc, f95, f_scores[::-1]


def entropy_of_expected_class(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_examples, num_classes]
    :return: array [num_examples}
    """
    mean_probs = np.mean(probs, axis=0)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=1)


def expected_entropy_class(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_examples, num_classes]
    :return: array [num_examples}
    """
    log_probs = -np.log(probs + epsilon)

    return np.mean(np.sum(probs * log_probs, axis=2), axis=0)


def ensemble_uncertainties_classification(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_examples, num_classes]
    :return: Dictionary of uncertaintties
    """
    mean_probs = np.mean(probs, axis=0)
    mean_lprobs = np.mean(np.log(probs + epsilon), axis=0)
    conf = np.max(mean_probs, axis=1)

    eoe = entropy_of_expected_class(probs, epsilon)
    exe = expected_entropy_class(probs, epsilon)

    mutual_info = eoe - exe

    epkl = -np.sum(mean_probs * mean_lprobs, axis=1) - exe

    uncertainty = {'confidence': conf,
                   'entropy_of_expected': eoe,
                   'expected_entropy': exe,
                   'mutual_information': mutual_info,
                   'epkl': epkl,
                   'reverse_mutual_information': epkl - mutual_info
                   }

    return uncertainty


def epkl_reg(preds, epsilon=1e-10):
    """
    preds: array [n_models, n_samples, 2] - mean and var along last axis.
    """
    means = preds[:, :, 0]
    vars = preds[:, :, 1] + epsilon
    logvars = np.log(vars)

    avg_means = np.mean(means, axis=0)
    avg_second_moments = np.mean(means * means + vars, axis=0)

    inv_vars = 1.0 / vars
    avg_inv_vars = np.mean(inv_vars, axis=0)
    mean_inv_var = inv_vars * means
    avg_mean_inv_var = np.mean(mean_inv_var, axis=0)
    avg_mean2_inv_var = np.mean(means * mean_inv_var + logvars, axis=0) + np.log(2 * np.pi)

    epkl = 0.5 * (avg_second_moments * avg_inv_vars - 2 * avg_means * avg_mean_inv_var + avg_mean2_inv_var)

    return epkl


def get_predictions(features_df, model):
    '''
    Calculates predictions on df features for specified model

    Return: array [num_samples x 2],
        where
            num_samples = number of rows in features_df
            2 = [mean, variance]

    '''
    return model.predict(features_df)


def get_all_predictions(features_df, models_list):
    '''
    Return: array [ensemble_size x num_samples x 2],
        where
            ensemble_size = number of models in models_list
            num_samples = number of rows in features_df
            2 = [mean, variance]
    '''
    all_preds = []
    for model in models_list:
        preds = np.asarray(get_predictions(features_df, model))
        all_preds.append(preds)
    return np.stack(all_preds, axis=0)


def ensemble_uncertainties_regression(preds):
    """
    preds: array [n_models, n_samples, 2] - last dim is mean, var
    """
    epkl = epkl_reg(preds)

    var_mean = np.var(preds[:, :, 0], axis=0)
    mean_var = np.mean(preds[:, :, 1], axis=0)

    uncertainty = {'tvar': var_mean + mean_var,
                   'mvar': mean_var,
                   'varm': var_mean,
                   'epkl': epkl}

    return uncertainty


def get_uncertainty_catboost(X_train_df, y_train_df, category_feature):
    # 불확실성 추정을 위한 앙상블 모델
    X_train, X_test, y_train, y_test = \
        train_test_split(X_train_df, y_train_df, test_size=0.2, random_state=42)

    ensemble_size = 3

    trained_models = []
    for seed in range(ensemble_size):
        model_cat = final_catboost(X_train, y_train, X_test, category_feature)

        print(f'\n Model index: {seed}\n')

        model_cat.fit(
            X_train,
            y_train,
            verbose=100,
            eval_set=(X_test, y_test))

        trained_models.append(model_cat)

    return trained_models


def get_uncertainty_feature(X_train_df, y_train_df, X_test_df, category_feature):
    trained_models = get_uncertainty_catboost(X_train_df, y_train_df, category_feature)
    all_preds = get_all_predictions(X_train_df, trained_models)
    all_uncertainty = ensemble_uncertainties_regression(all_preds)
    uncertainties = all_uncertainty['tvar']
    uncertainties_df = pd.DataFrame({'uncertainty': uncertainties})
    X_train_df.reset_index(inplace=True)
    X_train_df_new = pd.concat([X_train_df, uncertainties_df], axis=1)
    X_train_df_new = X_train_df_new.iloc[:, 1:]

    all_preds = get_all_predictions(X_test_df, trained_models)
    all_uncertainty = ensemble_uncertainties_regression(all_preds)
    uncertainties = all_uncertainty['tvar']
    uncertainties_df = pd.DataFrame({'uncertainty': uncertainties})
    X_test_df.reset_index(inplace=True)
    X_test_df_new = pd.concat([X_test_df, uncertainties_df], axis=1)
    X_test_df_new = X_test_df_new.iloc[:, 1:]

    return uncertainties, X_train_df_new, X_test_df_new


def quantile_score(y_true, y_pred, alpha):
    score = quantile_cost(x=y_true - y_pred, alpha=alpha)
    score = np.sum(score)
    return score


def quantile_cost(x, alpha):
    return (alpha - 1.0) * x * (x < 0) + alpha * x * (x >= 0)


def collect_prediction(X_train, y_train, X_test, y_test, estimator, alpha, model_name):
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print("{model_name} alpha = {alpha:.2f},score = {score:.1f}".format(model_name=model_name, alpha=alpha,
                                                                        score=quantile_score(y_test, y_pred,
                                                                                                         alpha)))

    return y_pred


def plot_result(X_train, y_train, X_test, y_test, y_pred, y_upper, y_lower):
    plt.plot(X_test, y_test, 'g:', label=u'test data')
    plt.plot(X_train, y_train, 'b.', markersize=10, label=u'Observations')
    plt.plot(X_test, y_pred, 'r-', label=u'Prediction')
    plt.plot(X_test, y_upper, 'k-')
    plt.plot(X_test, y_lower, 'k-')
    plt.fill(np.concatenate([X_test, X_test[::-1]]),
             np.concatenate([y_upper, y_lower[::-1]]),
             alpha=.5, fc='b', ec='None', label='90% prediction interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
