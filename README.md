<h1>Data pre-processing</h1>
Use *_processed_with_dummies(k-1).csv for regressions like OLS to avoid multicollinearity.
Otherwise for ML models use *_processed.csv or *_processed_with_dummies.csv.

Note that the first column is Id.
<pre>
from preprocessing import *
train_df, valid_df, test_df = get_processed_datasets(suffix='_with_dummies(k-1)')
</pre>

<h2>Splitting datasets</h2>
As the test.csv from Kaggle does not contain the target column SalePrice. We manually split the train.csv. We do not touch the validation dataset until the validation step.

Number of samples: 1021 for training, 439 for validation.

<h2>Feature transformation</h2>
<pre>
df['WasRemodeled'] = np.where(df.YearRemodAdd != df.YearBuilt, 'Y', 'N')
df['AgeWhenSold'] = df.YrSold - df.YearBuilt
df['TotalBath'] = df['FullBath'] + df['BsmtFullBath'] + df['HalfBath'] + df['BsmtHalfBath']
</pre>



<h2>Feature selection</h2>
We select features based on few criteria: assumptions from papers or surveys, correlation with SalePrice, (categorical) at least 2 groups with meaningful size.

<code>COLS_TO_SELECT = ['MSZoning', 'LotFrontage', 'LotArea', 'LotShape', 'Neighborhood',
                  'Condition1', 'BldgType', 'HouseStyle', 'TotalBsmtSF', 'Foundation', 'CentralAir',
                  'GrLivArea', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'TotalBath',
                  'GarageCars', 'GarageArea', 'WasRemodeled', 'AgeWhenSold']</code>

<h2>Null values</h2>
For the selected features, the only feature with null values in training dataset is LotFrontage. We fill it with 0.
In case there are unexpected null values appear in validation dataset or future data, we will use this strategy:

For numerical features, we fill it with the median. For categorical features, we fill it with the mode.

<h2>Outliners</h2>
We scan out features with values > 3std / < -3std and clip the values.