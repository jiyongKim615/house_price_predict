from train.preprocess import preprocess_missing_data
from utils.utils_preprocess import read_data

if __name__ == '__main__':
    train_df, test_df = read_data()
preprocess_missing_data

