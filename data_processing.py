from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np


def processing(train_data, test_data):
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

    # Remove outliers manually

    train_data = train_data.drop(train_data[(train_data['GrLivArea'] > 4000)
                                            & (train_data['SalePrice'] < 300000)].index).reset_index(drop=True)
    train_data = train_data.drop(train_data[(train_data['GarageCars'] > 3)
                                            & (train_data['SalePrice'] < 300000)].index).reset_index(drop=True)
    train_data = train_data.drop(train_data[(train_data['GarageArea'] > 1000)
                                            & (train_data['SalePrice'] < 300000)].index).reset_index(drop=True)

    # Imputing missing data, and encoding some categorical data with label encoder and one hot encoder

    # Drop missing data over 80%
    for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence'):
        train_data.drop([col], axis=1)
        test_data.drop([col], axis=1)

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'MasVnrType', 'Electrical', 'SaleType', 'MSZoning', 'Utilities',
                'Functional', 'Exterior2nd', 'Exterior1st', 'KitchenQual'):
        train_data[col] = train_data[col].fillna('None')
        train_data[col] = label_encoder.fit_transform(train_data[col])
        test_data[col] = test_data[col].fillna('None')
        test_data[col] = label_encoder.fit_transform(test_data[col])

    for col in (
            'CentralAir', 'Electrical', 'HeatingQC', 'Neighborhood', 'Heating', 'ExterQual', 'ExterCond',
            'SaleCondition', 'Street',
            'LotShape', 'LotConfig', 'LandSlope', 'LandContour', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Foundation', 'PavedDrive'):
        train_data[col] = label_encoder.fit_transform(train_data[col])
        test_data[col] = label_encoder.fit_transform(test_data[col])

    # Now, we select the numerical data from dataset
    # These numerical data include the data that we encoded (categorical of old)

    num_train = train_data.select_dtypes(include=[np.number])
    num_test = test_data.select_dtypes(include=[np.number])

    # Now, we impute the data for numerical ones

    num_train['LotFrontage'] = num_train['LotFrontage'].fillna(num_train['LotFrontage'].mean())
    num_train['GarageYrBlt'] = num_train['GarageYrBlt'].fillna(num_train['GarageYrBlt'].mean())
    num_train['MasVnrArea'] = num_train['MasVnrArea'].fillna(0)

    num_test['LotFrontage'] = num_test['LotFrontage'].fillna(num_test['LotFrontage'].mean())
    num_test['GarageYrBlt'] = num_test['GarageYrBlt'].fillna(num_test['GarageYrBlt'].mean())
    num_test['MasVnrArea'] = num_test['MasVnrArea'].fillna(0)
    num_test['BsmtFinSF1'] = num_test['BsmtFinSF1'].fillna(num_test['BsmtFinSF1'].mean())
    num_test['BsmtFinSF2'] = num_test['BsmtFinSF2'].fillna(num_test['BsmtFinSF2'].mean())
    num_test['BsmtUnfSF'] = num_test['BsmtUnfSF'].fillna(num_test['BsmtUnfSF'].mean())
    num_test['TotalBsmtSF'] = num_test['TotalBsmtSF'].fillna(num_test['TotalBsmtSF'].mean())
    num_test['BsmtFullBath'] = num_test['BsmtFullBath'].fillna(num_test['BsmtFullBath'].mean())
    num_test['BsmtHalfBath'] = num_test['BsmtHalfBath'].fillna(num_test['BsmtHalfBath'].mean())
    num_test['GarageYrBlt'] = num_test['GarageYrBlt'].fillna(num_test['GarageYrBlt'].mean())
    num_test['GarageCars'] = num_test['GarageCars'].fillna(num_test['GarageCars'].mean())
    num_test['GarageArea'] = num_test['GarageArea'].fillna(num_test['GarageArea'].mean())

    print(num_test.shape[1])

    return num_train, num_test
