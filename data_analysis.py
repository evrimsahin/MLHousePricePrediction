import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import rcParams


def visualization(df_train):
    print(df_train['SalePrice'].describe())  # We’re going to predict the SalePrice column ($ USD)

    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    rcParams['figure.figsize'] = 14, 8
    sns.distplot(df_train['SalePrice'], fit=norm)
    (mu, sigma) = norm.fit(df_train['SalePrice'])
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.title('Sale Prices')
    plt.xlabel('Sale Price')
    plt.ylabel('Probability')
    # plt.close()
    plt.show()
    # Most of the density lies between 100k and 250k, but there appears to be a lot of outliers on the pricier side.

    # -------------------------------- #

    # top 10 correlated features with the sale price:
    corr_matrix = df_train.corr()
    sns.heatmap(corr_matrix, vmax=.8, square=True)
    k = 10  # number of variables for heat map
    cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
    sns.heatmap(df_train[cols].corr().values.T, cbar=True, annot=True, square=True, yticklabels=cols.values,
                xticklabels=cols.values)
    # plt.close()
    plt.show()

    # Overall Quality vs Sale Price
    var = 'OverallQual'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000), s=32)
    plt.show()

    # Living Area vs Sale Price
    var = 'GrLivArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000), s=32)
    plt.show()

    # It makes sense that people would pay for the more living area.
    # What doesn't make sense is the two data points in the bottom-right of the plot.

    # Removing outliers manually (Two points in the bottom right)
    df_train = df_train.drop(df_train[(df_train['GrLivArea'] > 4000)
                                      & (df_train['SalePrice'] < 300000)].index).reset_index(drop=True)

    # After removing outliers, Living Area vs Sale Price
    var = 'GrLivArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000), s=32)
    plt.show()

    # GarageCars vs Sale Price
    var = 'GarageCars'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000), s=32)
    plt.show()

    # GarageArea vs Sale Price
    var = 'GarageArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000), s=32)
    plt.show()

    # Up to this point, we were exploring the data

    # Do we have missing data?

    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    missing_data = missing_data[missing_data.Total > 0]

    print(missing_data)

    # Then, we will impute the missing data

    ################################################################
