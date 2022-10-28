import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew  # for some statistics
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

fold = '/Users/jiyongkim/Documents/not_finish/house_price/house_price_regression_data/'
train = 'train.csv'
test = 'test.csv'


# 각 특징의 의미 파악
'''
다음은 데이터 설명 파일에서 찾을 수 있는 내용의 간략한 버전입니다.

SalePrice - 부동산의 판매 가격(달러)입니다. 이것은 예측하려는 대상 변수입니다.
MSSubClass: 건물 클래스
MSZoning: 일반 구역 분류
LotFrontage: 부동산에 연결된 거리의 선형 피트
LotArea: 로트 크기(제곱피트)
Street: 도로 접근 유형
Alley: 골목 접근 유형
LotShape: 속성의 일반적인 모양
LandContour: 속성의 평탄도
Utilities: 사용 가능한 유틸리티 유형
LotConfig: 로트 구성
LandSlope: 속성의 기울기
Neighborhood: Ames 시 경계 내의 물리적 위치
Condition1: 간선도로 또는 철도와 인접
Condition2: 주요 도로 또는 철도에 근접(초가 있는 경우)
BldgType: 주거 유형
HouseStyle: 주거 스타일
OverallQual: 전체 재료 및 마감 품질
OverallCond: 전체 상태 등급
YearBuilt: 원래 건설 날짜
YearRemodAdd: 리모델링 날짜
RoofStyle: 지붕 유형
RoofMatl: 지붕 재료
Exterior1st: 집의 외부 덮개
Exterior2nd: 주택의 외부 덮개(두 개 이상의 재료인 경우)
MasVnrType: 석조 베니어 유형
MasVnrArea: 석조 베니어판 면적(제곱피트)
ExterQual: 외장재 품질
ExterCond: 외장재의 현황
Foundation: 기초 유형
BsmtQual: 지하실 높이
BsmtCond: 지하실의 일반 상태
BsmtExposure: 파업 또는 정원 수준의 지하 벽
BsmtFinType1: 지하실 마감 면적의 품질
BsmtFinSF1: 유형 1 완성된 평방 피트
BsmtFinType2: 두 번째 완성 영역의 품질(있는 경우)
BsmtFinSF2: 유형 2 완성된 평방 피트
BsmtUnfSF: 지하실의 미완성 평방 피트
TotalBsmtSF: 지하 면적의 총 평방 피트
Heating: 난방 유형
HeatingQC: 난방 품질 및 상태
CentralAir: 중앙 에어컨
Electrical: 전기 시스템
1stFlrSF: 1층 평방피트
2ndFlrSF: 2층 평방피트
LowQualFinSF: 저품질 마감 평방 피트(모든 층)
GrLivArea: 지상(지상) 거실 면적 평방 피트
BsmtFullBath: 지하 전체 욕실
BsmtHalfBath: 지하 반 욕실
FullBath: 등급 이상의 전체 욕실
HalfBath: 등급 이상의 반 목욕
Bedroom: 지하층 이상의 침실 수
Kitchen: 주방 수
KitchenQual: 주방 품질
TotRmsAbvGrd: 등급 이상의 총 방(화장실 제외)
Functional: 홈 기능 등급
Fireplaces: 벽난로의 수
FireplaceQu: 벽난로 품질
GarageType: 차고 위치
GarageYrBlt: 차고가 건설된 해
GarageFinish: 차고의 인테리어 마감
GarageCars: 차고의 차고 크기
GarageArea: 평방 피트의 차고 크기
GarageQual: 차고 품질
GarageCond: 차고 상태
PavedDrive: 포장된 차도
WoodDeckSF: 평방 피트의 목재 데크 면적
OpenPorchSF: 평방 피트의 오픈 베란다 영역
EnclosedPorch: 제곱피트의 밀폐된 베란다 영역
3SsnPorch: 3계절 베란다 면적(제곱피트)
ScreenPorch: 스크린 베란다 면적(제곱피트)
PoolArea: 평방 피트의 수영장 면적
PoolQC: 수영장 품질
Fence: 울타리 품질
MiscFeature: 다른 범주에서 다루지 않는 기타 피처
MiscVal: 기타 기능의 $값
MoSold: 월 판매
YrSold: 판매 연도
SaleType: 판매 유형
SaleCondition : 판매조건
'''


'''
PoolQC(category): 수영장 품질 --> NA는 수영장이 없어서 발생될 수 있는 것 --> None으로 처리
MiscFeature(category): 다른 범주에서 다루지 않으므로 NA를 빈번한 값으로 처리
Alley(category): 골목접근유형 --> 빈번 값
Fence(category): 울타리 품질 --> None 처리
FireplaceQu(category): 벽난로 품질 --> None 처리
LotFrontage(numerous): 부동산에 연결된 거리 폭 -> 이웃집과 비슷할 것으로 판단 -> group & mean 적용
Garage{} --> 차고 관련 피처는 NA는 None
BsmtCond --> 상태 --> None 처리
BsmtQual --> None 처리
BsmtExposure --> 빈번 값
BsmtFinType1, BsmtFinType2 --> None
MasVnrType --> 유형 --> 빈번 값
###
MasVnrArea
MSZoning
Functional
BsmtHalfBath
BsmtFullBath
Utilities
GarageCars
KitchenQual
BsmtFinSF1
SaleType
BsmtFinSF2
BsmtUnfSF
TotalBsmtSF
Exterior2nd
Exterior1st
GarageArea
###
Electrical --> 학습 데이터만 있음 --> 빈번 값
'''


def read_data():
    train_df = pd.read_csv(fold + train)
    test_df = pd.read_csv(fold + test)
    return train_df, test_df


def plot_norm_skew_stats_target(df, target):
    sns.distplot(df[target], fit=norm)
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df[target])
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
               loc='best')
    plt.ylabel('Frequency')
    plt.title('Target distribution')

    # Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(df[target], plot=plt)
    plt.show()


def get_numerous_category_feature(df):
    category_feature = [col for col in df.columns if df[col].dtypes == "object"]
    int64_feature = [col for col in df.columns if df[col].dtypes == "int64"]
    float64_feature = [col for col in df.columns if df[col].dtypes == "float64"]
    return category_feature, int64_feature, float64_feature


def plot_corr(df, feature, target):
    fig, ax = plt.subplots()
    ax.scatter(x=df[feature], y=df[target])
    plt.ylabel(target, fontsize=13)
    plt.xlabel(feature, fontsize=13)
    plt.show()


def plot_all_corr(train_df):
    # 변수 상관성 분석
    # Correlation map to see how features are correlated with SalePrice
    corrmat = train_df.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)


def get_missing_data_percentage(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


def get_missing_data_remaining(all_data):
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
    return missing_data


def get_log_transform(df, target):
    # Target 변수의 로그변환
    # We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    df_copy = df.copy()
    df_copy[target] = np.log1p(df[target])

    # Check the new distribution
    sns.distplot(df_copy[target], fit=norm)

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df_copy[target])
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
               loc='best')
    plt.ylabel('Frequency')
    plt.title('Target distribution')

    # Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(df_copy[target], plot=plt)
    plt.show()

    return df_copy


def concat_train_test(train_df, test_df, target):
    n_train = train_df.shape[0]
    y_train = train_df[target].values
    all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
    all_data.drop([target], axis=1, inplace=True)
    print("all_data size is : {}".format(all_data.shape))

    return all_data, n_train, y_train


def fill_na_with_zero(all_data, feature):
    all_data[feature] = all_data[feature].fillna(0)
    return all_data


def fill_na_with_none(all_data, feature):
    all_data[feature] = all_data[feature].fillna("None")
    return all_data


def fill_na_with_mode(all_data, feature):
    all_data[feature] = all_data[feature].fillna(all_data[feature].mode()[0])
    return all_data


def fill_na_median_group_by_without_data_leakage(train_df, test_df, all_data, group_fe, agg_fe):
    train_df[agg_fe] = train_df.groupby(group_fe)[agg_fe].transform(
        lambda x: x.fillna(x.median()))
    test_df[agg_fe] = test_df.groupby(group_fe)[agg_fe].transform(
        lambda x: x.fillna(x.median()))

    all_data2 = pd.concat((train_df, test_df)).reset_index(drop=True)

    all_data[agg_fe] = all_data2[agg_fe]
    return all_data


def change_numerous_to_actual_cate(all_data, num_feature):
    # 실제로 범주형인 일부 숫자형 변수에 대해 변환 처리
    all_data[num_feature] = all_data[num_feature].apply(str)
    return all_data


def add_label_seq_label_encoding(all_data, cols):
    # 레이블 순서 집합에 정보를 포함할 수 있는 일부 범주형 변수 인코딩
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))
    # shape
    print('Shape all_data: {}'.format(all_data.shape))

    return all_data


def get_skewed_feature_stats(all_data):
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    return skewness


def box_cox_transform(all_data, skew_threshold, skew_lambda=0.15):
    skewness = get_skewed_feature_stats(all_data)
    skewness = skewness[abs(skewness) > skew_threshold]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
    skewed_features = skewness.index
    for feat in skewed_features:
        # all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], skew_lambda)

    # all_data[skewed_features] = np.log1p(all_data[skewed_features])

    return all_data


def get_dummy(all_data):
    # 더미 변수 생성
    all_data = pd.get_dummies(all_data)
    print(all_data.shape)
    return all_data


def get_train_test(all_data, ntrain):
    # 학습/테스트 데이터 재분리
    train_df = all_data[:ntrain]
    test_df = all_data[ntrain:]
    return train_df, test_df
