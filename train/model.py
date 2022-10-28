from utils.utils_train import *


def final_model_result(stacked_averaged_models, model_lgb, train, y_train, test):
    stacked_averaged_models.fit(train.values, y_train)
    stacked_train_pred = stacked_averaged_models.predict(train.values)
    stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    print(rmsle(y_train, stacked_train_pred))

    model_lgb.fit(train, y_train)
    lgb_train_pred = model_lgb.predict(train)
    lgb_pred = np.expm1(model_lgb.predict(test.values))
    print(rmsle(y_train, lgb_train_pred))

    print('RMSLE score on train data:')
    print(rmsle(y_train, stacked_train_pred * 0.70 + lgb_train_pred * 0.3))

    ensemble = stacked_pred * 0.70 + lgb_pred * 3

    return ensemble