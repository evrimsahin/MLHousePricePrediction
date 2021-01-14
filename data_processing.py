from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd


def processing(train_data, test_data):
    labelencoder = LabelEncoder()
    enc = OneHotEncoder(handle_unknown='ignore')

    train_data = train_data.drop(train_data[(train_data['GrLivArea'] > 4000)
                                            & (train_data['SalePrice'] < 300000)].index).reset_index(drop=True)
    train_data = train_data.drop(train_data[(train_data['GarageCars'] > 3)
                                            & (train_data['SalePrice'] < 300000)].index).reset_index(drop=True)

    train_data = train_data.drop(train_data[(train_data['GarageArea'] > 1000)
                                            & (train_data['SalePrice'] < 300000)].index).reset_index(drop=True)

    """
    train_data['FireplaceQu'].replace(to_replace=np.nan, value="TA", inplace=True)
    train_data['FireplaceQu'] = labelencoder.fit_transform(train_data['FireplaceQu'])
    print(train_data['FireplaceQu'])"""

    train_data['FireplaceQu'].replace(to_replace=np.nan, value="TA", inplace=True)
    train_data['FireplaceQu'] = labelencoder.fit_transform(train_data['FireplaceQu'])
    test_data['FireplaceQu'].replace(to_replace=np.nan, value="TA", inplace=True)
    test_data['FireplaceQu'] = labelencoder.fit_transform(test_data['FireplaceQu'])

    train_data['ExterQual'].replace(to_replace=np.nan, value="TA", inplace=True)
    train_data['ExterQual'] = labelencoder.fit_transform(train_data['ExterQual'])
    test_data['ExterQual'].replace(to_replace=np.nan, value="TA", inplace=True)
    test_data['ExterQual'] = labelencoder.fit_transform(test_data['ExterQual'])

    train_data['BsmtQual'].replace(to_replace=np.nan, value="TA", inplace=True)
    train_data['BsmtQual'] = labelencoder.fit_transform(train_data['BsmtQual'])
    test_data['BsmtQual'].replace(to_replace=np.nan, value="TA", inplace=True)
    test_data['BsmtQual'] = labelencoder.fit_transform(test_data['BsmtQual'])

    train_data['BsmtCond'].replace(to_replace=np.nan, value="TA", inplace=True)
    train_data['BsmtCond'] = labelencoder.fit_transform(train_data['BsmtCond'])
    test_data['BsmtCond'].replace(to_replace=np.nan, value="TA", inplace=True)
    test_data['BsmtCond'] = labelencoder.fit_transform(test_data['BsmtCond'])

    train_data['KitchenQual'].replace(to_replace=np.nan, value="TA", inplace=True)
    train_data['KitchenQual'] = labelencoder.fit_transform(train_data['KitchenQual'])
    test_data['KitchenQual'].replace(to_replace=np.nan, value="TA", inplace=True)
    test_data['KitchenQual'] = labelencoder.fit_transform(test_data['KitchenQual'])

    """train_data['BsmtExposure'].replace(to_replace=np.nan, value="Av", inplace=True)
    train_data['BsmtExposure'] = labelencoder.fit_transform(train_data['BsmtExposure'])
    test_data['BsmtExposure'].replace(to_replace=np.nan, value="Av", inplace=True)
    test_data['BsmtExposure'] = labelencoder.fit_transform(test_data['BsmtExposure'])
    
    train_data['BsmtFinType1'].replace(to_replace=np.nan, value="Rec", inplace=True)
    train_data['BsmtFinType1'] = labelencoder.fit_transform(train_data['BsmtFinType1'])
    test_data['BsmtFinType1'].replace(to_replace=np.nan, value="Rec", inplace=True)
    test_data['BsmtFinType1'] = labelencoder.fit_transform(test_data['BsmtFinType1'])
    
    train_data['BsmtFinType2'].replace(to_replace=np.nan, value="Rec", inplace=True)
    train_data['BsmtFinType2'] = labelencoder.fit_transform(train_data['BsmtFinType2'])
    test_data['BsmtFinType2'].replace(to_replace=np.nan, value="Rec", inplace=True)
    test_data['BsmtFinType2'] = labelencoder.fit_transform(test_data['BsmtFinType2'])
    
    train_data['HeatingQC'].replace(to_replace=np.nan, value="TA", inplace=True)
    train_data['HeatingQC'] = labelencoder.fit_transform(train_data['HeatingQC'])
    test_data['HeatingQC'].replace(to_replace=np.nan, value="TA", inplace=True)
    test_data['HeatingQC'] = labelencoder.fit_transform(test_data['HeatingQC'])"""

    """train_data['CentralAir'].replace(to_replace=np.nan, value=str(train_data['CentralAir'].mode()), inplace=True)
    train_data['CentralAir'] = labelencoder.fit_transform(train_data['CentralAir'])
    test_data['CentralAir'].replace(to_replace=np.nan, value=str(train_data['CentralAir'].mode()), inplace=True)
    test_data['CentralAir'] = labelencoder.fit_transform(test_data['CentralAir'])"""

    """train_data['Electrical'].replace(to_replace=np.nan, value="FuseA", inplace=True)
    train_data['Electrical'] = labelencoder.fit_transform(train_data['Electrical'])
    test_data['Electrical'].replace(to_replace=np.nan, value="FuseA", inplace=True)
    test_data['Electrical'] = labelencoder.fit_transform(test_data['Electrical'])"""

    """train_data['GarageQual'].replace(to_replace=np.nan, value="TA", inplace=True)
    train_data['GarageQual'] = labelencoder.fit_transform(train_data['GarageQual'])
    test_data['GarageQual'].replace(to_replace=np.nan, value="TA", inplace=True)
    test_data['GarageQual'] = labelencoder.fit_transform(test_data['GarageQual'])
    
    
    train_data['GarageCond'].replace(to_replace=np.nan, value="TA", inplace=True)
    train_data['GarageCond'] = labelencoder.fit_transform(train_data['GarageCond'])
    test_data['GarageCond'].replace(to_replace=np.nan, value="TA", inplace=True)
    test_data['GarageCond'] = labelencoder.fit_transform(test_data['GarageCond'])
    
    train_data['PoolQC'].replace(to_replace=np.nan, value="TA", inplace=True)
    train_data['PoolQC'] = labelencoder.fit_transform(train_data['PoolQC'])
    test_data['PoolQC'].replace(to_replace=np.nan, value="TA", inplace=True)
    test_data['PoolQC'] = labelencoder.fit_transform(test_data['PoolQC'])"""

    """train_data['Fence'].replace(to_replace=np.nan, value="GdWo", inplace=True)
    train_data['Fence'] = labelencoder.fit_transform(train_data['Fence'])
    test_data['Fence'].replace(to_replace=np.nan, value="GdWo", inplace=True)
    test_data['Fence'] = labelencoder.fit_transform(test_data['Fence'])"""

    train_data['Neighborhood'] = pd.DataFrame(enc.fit_transform(train_data[['Neighborhood']]).toarray())
    test_data['Neighborhood'] = pd.DataFrame(enc.fit_transform(test_data[['Neighborhood']]).toarray())

    train_data['Heating'] = pd.DataFrame(enc.fit_transform(train_data[['Heating']]).toarray())
    test_data['Heating'] = pd.DataFrame(enc.fit_transform(test_data[['Heating']]).toarray())

    train_data['SaleType'].replace(to_replace=np.nan, value="None", inplace=True)
    train_data['SaleType'] = pd.DataFrame(enc.fit_transform(train_data[['SaleType']]).toarray())
    test_data['SaleType'].replace(to_replace=np.nan, value="None", inplace=True)
    test_data['SaleType'] = pd.DataFrame(enc.fit_transform(test_data[['SaleType']]).toarray())

    train_data['SaleCondition'] = pd.DataFrame(enc.fit_transform(train_data[['SaleCondition']]).toarray())
    test_data['SaleCondition'] = pd.DataFrame(enc.fit_transform(test_data[['SaleCondition']]).toarray())

    num_train = train_data.select_dtypes(include=[np.number])
    num_test = test_data.select_dtypes(include=[np.number])
    num_train['LotFrontage'].replace(to_replace=np.nan, value=int(num_train['LotFrontage'].mean()), inplace=True)
    num_train['GarageYrBlt'].replace(to_replace=np.nan, value=int(num_train['GarageYrBlt'].mean()), inplace=True)
    num_train['MasVnrArea'].replace(to_replace=np.nan, value=0, inplace=True)

    num_test['LotFrontage'].replace(to_replace=np.nan, value=int(num_test['LotFrontage'].mean()), inplace=True)
    num_test['GarageYrBlt'].replace(to_replace=np.nan, value=int(num_test['GarageYrBlt'].mean()), inplace=True)
    num_test['MasVnrArea'].replace(to_replace=np.nan, value=0, inplace=True)
    num_test['BsmtFinSF1'].replace(to_replace=np.nan, value=int(num_test['BsmtFinSF1'].mean()), inplace=True)
    num_test['BsmtFinSF2'].replace(to_replace=np.nan, value=int(num_test['BsmtFinSF2'].mean()), inplace=True)
    num_test['BsmtUnfSF'].replace(to_replace=np.nan, value=int(num_test['BsmtUnfSF'].mean()), inplace=True)
    num_test['TotalBsmtSF'].replace(to_replace=np.nan, value=int(num_test['TotalBsmtSF'].mean()), inplace=True)
    num_test['BsmtFullBath'].replace(to_replace=np.nan, value=int(num_test['BsmtFullBath'].mean()), inplace=True)
    num_test['BsmtHalfBath'].replace(to_replace=np.nan, value=int(num_test['BsmtHalfBath'].mean()), inplace=True)
    num_test['GarageCars'].replace(to_replace=np.nan, value=int(num_test['GarageCars'].mean()), inplace=True)
    num_test['GarageArea'].replace(to_replace=np.nan, value=int(num_test['GarageArea'].mean()), inplace=True)

    return num_train, num_test