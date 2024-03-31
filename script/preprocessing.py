import pandas as pd
import numpy as np
DATA_PATH = "../data/"
PROCESSED_DATA_PATH = "../processed_data/"

COLS_TO_SELECT = ['MSZoning', 'LotFrontage', 'LotArea', 'LotShape', 'Neighborhood',
                  'Condition1', 'BldgType', 'HouseStyle', 'TotalBsmtSF', 'Foundation', 'CentralAir',
                  'GrLivArea', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'TotalBath',
                  'GarageCars', 'GarageArea', 'WasRemodeled', 'AgeWhenSold']

LIMITS = {
    'LotFrontage': (0, 150),
    'LotArea': (0, 30000),
    'TotalBsmtSF': (0, 2000),
    'GrLivArea': (0, 3000),
    'BedroomAbvGr': (1, 5),
    'TotRmsAbvGrd': (2, 10),
    'Fireplaces': (0, 2),
    'TotalBath': (1, 4),
    'GarageArea': (0, 1000),
    'AgeWhenSold': (0, 100),
}

def get_datasets(split_ratio=0.7, seed=0):
    dataset_df = pd.read_csv(DATA_PATH+"train.csv")
    dataset_df = dataset_df.sample(frac=1, random_state=seed)
    N_train = int(len(dataset_df) * split_ratio)
    train_df = dataset_df[:N_train]
    valid_df = dataset_df[N_train:] # for model evaluation
    test_df = pd.read_csv(DATA_PATH+"test.csv") # for Kaggle submission, no SalePrice
    return train_df, valid_df, test_df

def get_processed_datasets(suffix=''):
    train_df = pd.read_csv(PROCESSED_DATA_PATH+'train_processed'+suffix+'.csv', index_col=0, keep_default_na=False)
    valid_df = pd.read_csv(PROCESSED_DATA_PATH+'valid_processed'+suffix+'.csv', index_col=0, keep_default_na=False)
    test_df = pd.read_csv(PROCESSED_DATA_PATH+'test_processed'+suffix+'.csv', index_col=0, keep_default_na=False)
    return train_df, valid_df, test_df

def get_features(df):
    numeric_features = set(df.select_dtypes(['float64', 'int64']).columns)
    numeric_features = numeric_features.difference(set(['SalePrice']))
    category_features = set(df.select_dtypes(['O']).columns)
    return list(numeric_features), list(category_features)

# TODO: filter outliers from train_df
def preprocess_dataset(df, is_testing=False):
    df = df.set_index('Id')
    df['WasRemodeled'] = np.where(df.YearRemodAdd != df.YearBuilt, 'Y', 'N')
    df['AgeWhenSold'] = df.YrSold - df.YearBuilt
    df['TotalBath'] = df['FullBath'] + df['BsmtFullBath'] + df['HalfBath'] + df['BsmtHalfBath']

    df = df[COLS_TO_SELECT] if is_testing else df[COLS_TO_SELECT+['SalePrice']]

    # handle null values
    df['LotFrontage'] = df.LotFrontage.fillna(0.)
    numeric_features, category_features = get_features(df)
    # in case unseen data has unexpected null values
    for c in numeric_features+category_features:
        if c in numeric_features:
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode()[0])

    for k, (mini, maxi) in LIMITS.items():
        df[k] = df[k].clip(mini, maxi)

    if not is_testing:
        df = pd.concat([df.drop(columns='SalePrice'), df['SalePrice']], axis=1)
    return df

def replace_category_with_dummy_datasets(train_df, valid_df, test_df, to_drop):
    _, train_category_features = get_features(train_df)
    mode = train_df[train_category_features].mode().iloc[0]

    def replace_dummy(df, mode, to_print=False):
        _, category_features = get_features(df)
        df = pd.concat([pd.get_dummies(df[category_features], prefix=category_features), df], axis=1)
        df = df.drop(columns=category_features)
        df = df.astype(np.float64)
        # drop each most frequent category to avoid multicollinearity, relevant for regression models
        if to_drop:
            cols = [c for c in df.columns if '_' in c]
            dropping = []
            for c in cols:
                feature, category = c.split('_')
                if category in ['Y', 'N']:
                    if category=='N':
                        dropping.append(c)
                else:
                    if category==mode[feature]:
                        dropping.append(c)
            df = df.drop(columns=dropping)
            if to_print:
                print('dropped:')
                print('\n'.join(dropping))
        return df
    train_df = replace_dummy(train_df, mode, to_print=True)
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
