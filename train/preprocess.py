from utils.utils_preprocess import *


def preprocess_missing_data(all_data, train_df, test_df):
    all_data = fill_na_with_none(all_data, 'PoolQC')
    all_data = fill_na_with_mode(all_data, 'MiscFeature')
    all_data = fill_na_with_none(all_data, 'Alley')
    all_data = fill_na_with_none(all_data, 'Fence')
    all_data = fill_na_with_none(all_data, 'FireplaceQu')
    all_data = \
        fill_na_median_group_by_without_data_leakage(train_df, test_df, all_data, 'Neighborhood', 'LotFrontage')

    garlst = ['GarageYrBlt', 'GarageQual', 'GarageFinish', 'GarageCond', 'GarageType', 'GarageCars']
    for gar in garlst:
        all_data = fill_na_with_zero(all_data, gar)

    all_data = fill_na_with_none(all_data, 'BsmtCond')
    all_data = fill_na_with_none(all_data, 'BsmtQual')
    all_data = fill_na_with_mode(all_data, 'BsmtExposure')
    all_data = fill_na_with_none(all_data, 'BsmtFinType1')
    all_data = fill_na_with_none(all_data, 'BsmtFinType2')
    all_data = fill_na_with_mode(all_data, 'MasVnrType')
    all_data = fill_na_with_zero(all_data, 'MasVnrArea')
    # NA 개수가 적어서 최빈값으로 대체
    na_small_lst = ['MSZoning', 'Functional',
                    'BsmtHalfBath', 'BsmtFullBath', 'Utilities',
                    'KitchenQual',
                    'SaleType', 'Exterior2nd', 'Exterior1st', 'GarageArea']
    for mode_na in na_small_lst:
        all_data = fill_na_with_mode(all_data, mode_na)

    # 지하실 관련해서 0처리
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data = fill_na_with_zero(all_data, col)

    all_data = fill_na_with_mode(all_data, 'Electrical')

    return all_data


def preprocess_feature_encoding(all_data):
    # 실제 범주형의 성격을 가진 일부 숫자형 변수에 대한 변환 처리
    all_data = change_numerous_to_actual_cate(all_data, 'MSSubClass')
    all_data = change_numerous_to_actual_cate(all_data, 'OverallCond')
    all_data = change_numerous_to_actual_cate(all_data, 'YrSold')
    all_data = change_numerous_to_actual_cate(all_data, 'MoSold')

    # 레이블 순서 정보를 포함할 수 있는 일부 범주형 변수 인코딩
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')

    all_data = add_label_seq_label_encoding(all_data, cols)

    return all_data


def preprocess_skew_features(all_data, skew_threshold, skew_lambda=0.15):
    # check
    skewness = get_skewed_feature_stats(all_data)
    print(skewness)
    # Box Cox 변환
    all_data = box_cox_transform(all_data, skew_threshold, skew_lambda=skew_lambda)
    return all_data


