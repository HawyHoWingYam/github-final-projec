import pandas as pd
import numpy as np
DATA_PATH = "../data/"
PROCESSED_DATA_PATH = "../processed_data/"
COLS_TO_REMOVE = ['LowQualFinSF', 'MiscFeature', 'YearBuilt', 'YrSold', 'PoolQC', 'PoolArea', 'GarageYrBlt', 'YearRemodAdd', 'BsmtUnfSF']
FILLNA_DICT = {
    'MasVnrArea': 0.,
    'Electrical': 'SBrkr',
    'LotFrontage': 0.,
    'BsmtQual': 'NA',
    'BsmtCond': 'NA',
    'BsmtExposure': 'NA',
    'BsmtFinType1': 'NA',
    'BsmtFinType2': 'NA',
    'GarageFinish': 'NA',
    'GarageQual': 'NA',
    'GarageCond': 'NA',
    'GarageType': 'NA',
    'FireplaceQu': 'NA',
    'MasVnrType': 'NA',
    'Fence': 'NA',
    'Alley': 'NA',
}

def get_datasets(split_ratio=0.7, seed=0):
    dataset_df = pd.read_csv(DATA_PATH+"train.csv")
    dataset_df = dataset_df.sample(frac=1, random_state=seed)
    N_train = int(len(dataset_df) * split_ratio)
    train_df = dataset_df[:N_train]
    valid_df = dataset_df[N_train:] # for model evaluation
    test_df = pd.read_csv(DATA_PATH+"test.csv") # for Kaggle submission, no SalePrice
    return train_df, valid_df, test_df

def get_processed_datasets():
    train_df = pd.read_csv(PROCESSED_DATA_PATH+'train_processed.csv', index_col=0, keep_default_na=False)
    valid_df = pd.read_csv(PROCESSED_DATA_PATH+'valid_processed.csv', index_col=0, keep_default_na=False)
    test_df = pd.read_csv(PROCESSED_DATA_PATH+'test_processed.csv', index_col=0, keep_default_na=False)
    return train_df, valid_df, test_df

def get_features(df):
    numeric_features = set(df.select_dtypes(['float64', 'int64']).columns)
    numeric_features = numeric_features.difference(set(['SalePrice']))
    category_features = set(df.select_dtypes(['O']).columns)
    return list(numeric_features), list(category_features)

# TODO: filter outliers from train_df
def preprocess_dataset(df, is_testing=False):
    df = df.set_index('Id')
    df['MSSubClass'] = 'Class ' + df.MSSubClass.astype(str)
    df['WasRemodeled'] = np.where(df.YearRemodAdd != df.YearBuilt, 'Y', 'N')
    df['AgeWhenSold'] = df.YrSold - df.YearBuilt

    df = df.drop(columns=COLS_TO_REMOVE)

    # handle null values
    df = df.fillna(FILLNA_DICT)
    numeric_features, category_features = get_features(df)
    # in case unseen data has unexpected null values
    for c in numeric_features+category_features:
        if c in numeric_features:
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode()[0])

    rating_map1 = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': 0}
    rating_map2 = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
    lst = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
    
    df[lst] = df[lst].replace(rating_map1)
    df['BsmtExposure'] = df.BsmtExposure.replace(rating_map2)

    if not is_testing:
        df = pd.concat([df.drop(columns='SalePrice'), df['SalePrice']], axis=1)
    return df

def replace_category_with_dummy_datasets(train_df, valid_df, test_df):
    _, train_category_features = get_features(train_df)
    mode = train_df[train_category_features].mode().iloc[0]

    def replace_dummy(df, mode):
        _, category_features = get_features(df)
        df = pd.concat([pd.get_dummies(df[category_features], prefix=category_features), df], axis=1)
        df = df.drop(columns=category_features)
        df = df.astype(np.float64)
        # drop each first category to avoid multicollinearity
        cols = [c for c in df.columns if '_' in c]
        for c in cols:
            feature, category = c.split('_')
            if category in ['Y', 'N']:
                if category=='N':
                    df = df.drop(columns=c)
            else:
                if category==mode[feature]:
                    df = df.drop(columns=c)
        return df
    train_df = replace_dummy(train_df, mode)
    valid_df = replace_dummy(valid_df, mode)
    test_df = replace_dummy(test_df, mode)

    train_features = train_df.columns[:-1]

    # add categories that appeared in train_df but not valid_df / test_df
    valid_df[train_features[~train_features.isin(valid_df.columns)]] = 0.
    test_df[train_features[~train_features.isin(test_df.columns)]] = 0.

    # reorder columns and drop categories that did not appear in train_df
    valid_df = valid_df[list(train_features)+['SalePrice']]
    test_df = test_df[train_features]
    return train_df, valid_df, test_df
