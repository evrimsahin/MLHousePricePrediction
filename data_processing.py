from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd


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
    train_data.drop(['PoolQC'], axis=1)
    train_data.drop(['MiscFeature'], axis=1)
    train_data.drop(['Alley'], axis=1)
    train_data.drop(['Fence'], axis=1)

    test_data.drop(['PoolQC'], axis=1)
    test_data.drop(['MiscFeature'], axis=1)
    test_data.drop(['Alley'], axis=1)
    test_data.drop(['Fence'], axis=1)

    train_data["FireplaceQu"] = train_data["FireplaceQu"].fillna("None")
    train_data['FireplaceQu'] = label_encoder.fit_transform(train_data['FireplaceQu'])
    test_data["FireplaceQu"] = test_data["FireplaceQu"].fillna("None")
    test_data['FireplaceQu'] = label_encoder.fit_transform(test_data['FireplaceQu'])

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        train_data[col] = train_data[col].fillna('None')
        train_data[col] = label_encoder.fit_transform(train_data[col])
        test_data[col] = test_data[col].fillna('None')
        test_data[col] = label_encoder.fit_transform(test_data[col])

    train_data["BsmtExposure"] = train_data["BsmtExposure"].fillna("None")
    train_data['BsmtExposure'] = label_encoder.fit_transform(train_data['BsmtExposure'])
    test_data["BsmtExposure"] = test_data["BsmtExposure"].fillna("None")
    test_data['BsmtExposure'] = label_encoder.fit_transform(test_data['BsmtExposure'])

    train_data["BsmtFinType1"] = train_data["BsmtFinType1"].fillna("None")
    train_data['BsmtFinType1'] = label_encoder.fit_transform(train_data['BsmtFinType1'])
    test_data["BsmtFinType1"] = test_data["BsmtFinType1"].fillna("None")
    test_data['BsmtFinType1'] = label_encoder.fit_transform(test_data['BsmtFinType1'])

    train_data["BsmtFinType2"] = train_data["BsmtFinType2"].fillna("None")
    train_data['BsmtFinType2'] = label_encoder.fit_transform(train_data['BsmtFinType2'])
    test_data["BsmtFinType2"] = test_data["BsmtFinType2"].fillna("None")
    test_data['BsmtFinType2'] = label_encoder.fit_transform(test_data['BsmtFinType2'])

    train_data["BsmtCond"] = train_data["BsmtCond"].fillna("None")
    train_data['BsmtCond'] = label_encoder.fit_transform(train_data['BsmtCond'])
    test_data["BsmtCond"] = test_data["BsmtCond"].fillna("None")
    test_data['BsmtCond'] = label_encoder.fit_transform(test_data['BsmtCond'])

    train_data["BsmtQual"] = train_data["BsmtQual"].fillna("None")
    train_data['BsmtQual'] = label_encoder.fit_transform(train_data['BsmtQual'])
    test_data["BsmtQual"] = test_data["BsmtQual"].fillna("None")
    test_data['BsmtQual'] = label_encoder.fit_transform(test_data['BsmtQual'])

    train_data["MasVnrType"] = train_data["MasVnrType"].fillna("None")
    train_data['MasVnrType'] = label_encoder.fit_transform(train_data['MasVnrType'])
    test_data["MasVnrType"] = test_data["MasVnrType"].fillna("None")
    test_data['MasVnrType'] = label_encoder.fit_transform(test_data['MasVnrType'])

    train_data["Electrical"] = train_data["Electrical"].fillna("None")
    train_data['Electrical'] = label_encoder.fit_transform(train_data['Electrical'])
    test_data["Electrical"] = test_data["Electrical"].fillna("None")
    test_data['Electrical'] = label_encoder.fit_transform(test_data['Electrical'])

    train_data["SaleType"] = train_data["SaleType"].fillna("None")
    train_data['SaleType'] = pd.DataFrame(one_hot_encoder.fit_transform(train_data[['SaleType']]).toarray())
    test_data['SaleType'].replace(to_replace=np.nan, value="None", inplace=True)
    test_data['SaleType'] = pd.DataFrame(one_hot_encoder.fit_transform(test_data[['SaleType']]).toarray())

    test_data['MSZoning'] = test_data['MSZoning'].fillna('None')
    test_data['MSZoning'] = label_encoder.fit_transform(test_data['MSZoning'])
    train_data['MSZoning'] = label_encoder.fit_transform(train_data['MSZoning'])

    test_data['Utilities'] = test_data['Utilities'].fillna(test_data['Utilities'].mode()[0])
    test_data['Utilities'] = pd.DataFrame(one_hot_encoder.fit_transform(test_data[['Utilities']]).toarray())
    train_data['Utilities'] = pd.DataFrame(one_hot_encoder.fit_transform(train_data[['Utilities']]).toarray())

    test_data['Functional'] = test_data['Functional'].fillna(test_data['Functional'].mode()[0])
    test_data['Functional'] = pd.DataFrame(one_hot_encoder.fit_transform(test_data[['Functional']]).toarray())
    train_data['Functional'] = pd.DataFrame(one_hot_encoder.fit_transform(train_data[['Functional']]).toarray())

    test_data['Exterior2nd'] = test_data['Exterior2nd'].fillna(test_data['Exterior2nd'].mode()[0])
    test_data['Exterior2nd'] = pd.DataFrame(one_hot_encoder.fit_transform(test_data[['Exterior2nd']]).toarray())
    train_data['Exterior2nd'] = pd.DataFrame(one_hot_encoder.fit_transform(train_data[['Exterior2nd']]).toarray())

    test_data['Exterior1st'] = test_data['Exterior1st'].fillna(test_data['Exterior1st'].mode()[0])
    test_data['Exterior1st'] = pd.DataFrame(one_hot_encoder.fit_transform(test_data[['Exterior1st']]).toarray())
    train_data['Exterior1st'] = pd.DataFrame(one_hot_encoder.fit_transform(train_data[['Exterior1st']]).toarray())

    test_data["KitchenQual"] = test_data["KitchenQual"].fillna("None")
    test_data['KitchenQual'] = label_encoder.fit_transform(test_data['KitchenQual'])
    train_data['KitchenQual'] = label_encoder.fit_transform(train_data['KitchenQual'])

    train_data['CentralAir'] = label_encoder.fit_transform(train_data['CentralAir'])
    test_data['CentralAir'] = label_encoder.fit_transform(test_data['CentralAir'])

    train_data['Electrical'] = label_encoder.fit_transform(train_data['Electrical'])
    test_data['Electrical'] = label_encoder.fit_transform(test_data['Electrical'])

    train_data['HeatingQC'] = label_encoder.fit_transform(train_data['HeatingQC'])
    test_data['HeatingQC'] = label_encoder.fit_transform(test_data['HeatingQC'])

    train_data['Neighborhood'] = pd.DataFrame(one_hot_encoder.fit_transform(train_data[['Neighborhood']]).toarray())
    test_data['Neighborhood'] = pd.DataFrame(one_hot_encoder.fit_transform(test_data[['Neighborhood']]).toarray())

    train_data['Heating'] = pd.DataFrame(one_hot_encoder.fit_transform(train_data[['Heating']]).toarray())
    test_data['Heating'] = pd.DataFrame(one_hot_encoder.fit_transform(test_data[['Heating']]).toarray())

    train_data['ExterQual'] = label_encoder.fit_transform(train_data['ExterQual'])
    test_data['ExterQual'] = label_encoder.fit_transform(test_data['ExterQual'])

    train_data['SaleCondition'] = pd.DataFrame(one_hot_encoder.fit_transform(train_data[['SaleCondition']]).toarray())
    test_data['SaleCondition'] = pd.DataFrame(one_hot_encoder.fit_transform(test_data[['SaleCondition']]).toarray())

    train_data['Street'] = label_encoder.fit_transform(train_data['Street'])
    test_data['Street'] = label_encoder.fit_transform(test_data['Street'])

    train_data['LotShape'] = label_encoder.fit_transform(train_data['LotShape'])
    test_data['LotShape'] = label_encoder.fit_transform(test_data['LotShape'])

    # Now, we select the numerical data from dataset
    # These numerical data include the data that we encoded (categorical of old)

    num_train = train_data.select_dtypes(include=[np.number])
    num_test = test_data.select_dtypes(include=[np.number])

    # Now, we impute the data for numerical ones

    num_train['LotFrontage'].replace(to_replace=np.nan, value=int(num_train['LotFrontage'].mean()), inplace=True)
    num_train['GarageYrBlt'].replace(to_replace=np.nan, value=int(num_train['GarageYrBlt'].mean()), inplace=True)
    num_train['MasVnrArea'].replace(to_replace=np.nan, value=0, inplace=True)

    # Burada aynılarını neden train için yapmadık?

    num_test['LotFrontage'].replace(to_replace=np.nan, value=int(num_test['LotFrontage'].mean()), inplace=True)
    num_test['MasVnrArea'].replace(to_replace=np.nan, value=0, inplace=True)
    num_test['BsmtFinSF1'].replace(to_replace=np.nan, value=int(num_test['BsmtFinSF1'].mean()), inplace=True)
    num_test['BsmtFinSF2'].replace(to_replace=np.nan, value=int(num_test['BsmtFinSF2'].mean()), inplace=True)
    num_test['BsmtUnfSF'].replace(to_replace=np.nan, value=int(num_test['BsmtUnfSF'].mean()), inplace=True)
    num_test['TotalBsmtSF'].replace(to_replace=np.nan, value=int(num_test['TotalBsmtSF'].mean()), inplace=True)
    num_test['BsmtFullBath'].replace(to_replace=np.nan, value=int(num_test['BsmtFullBath'].mean()), inplace=True)
    num_test['BsmtHalfBath'].replace(to_replace=np.nan, value=int(num_test['BsmtHalfBath'].mean()), inplace=True)
    num_test['GarageYrBlt'].replace(to_replace=np.nan, value=int(num_test['GarageYrBlt'].mean()), inplace=True)
    num_test['GarageCars'].replace(to_replace=np.nan, value=int(num_test['GarageCars'].mean()), inplace=True)
    num_test['GarageArea'].replace(to_replace=np.nan, value=int(num_test['GarageArea'].mean()), inplace=True)

    return num_train, num_test
