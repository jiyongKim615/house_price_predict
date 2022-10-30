from utils.utils_preprocess import *
from utils.utils_train import get_outlier_label_by_ecod


def get_final_train_test():
    # 데이터 불러오기
    train_df, test_df = read_data()
    test_df_id = test_df['Id']
    train_df.drop('Id', axis=1, inplace=True)
    test_df.drop('Id', axis=1, inplace=True)
    train_df_target_log_transform = get_log_transform(train_df, 'SalePrice')
    # 학습/테스트 합쳐서 일괄 처리
    all_data, n_train, y_train_df = \
        concat_train_test(train_df_target_log_transform, test_df, 'SalePrice')
    # 결측값 처리
    all_data = preprocess_missing_data(all_data, train_df, test_df)
    # 추가 FE
    all_data = preprocess_feature_encoding(all_data)
    # 피처 추가
    all_data['TotalSF'] = \
        all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    # skew 처리
    all_data = preprocess_skew_features(all_data, skew_threshold=0.75, skew_lambda=0.15)
    # 더미변수 생성
    all_data = get_dummy(all_data)
    # 학습/테스트 분리
    train_df, test_df_no_target = get_train_test(all_data, n_train)
    # 이상값 처리
    train_df, no_outlier_train_df, no_outlier_y_train_df, test_df_no_target = \
        get_outlier_label_by_ecod(train_df, test_df_no_target, y_train_df)

    feature_cols = no_outlier_train_df.columns.tolist()

    X_train = no_outlier_train_df.copy()
    X_test = test_df_no_target[feature_cols]
    y_train = no_outlier_y_train_df.copy()
    return X_train, X_test, y_train, test_df_id


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


