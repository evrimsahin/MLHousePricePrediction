import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm


def visualization(df_train):
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

    # Next, let’s have a look at the greater living area (square feet) against the sale price:

    # Exploring the data - Sale Price vs GrLivArea
    data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
    data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000), s=32)

    # You might’ve expected that larger living area should mean a higher price.
    # This chart shows you’re generally correct. But what are those 2–3 “cheap” houses offering huge living area?

    corr_matrix = df_train.corr()
   # f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_matrix, vmax=.8, square=True)

    # Let’s have a more general view on the top 10 correlated features with the sale price:
    k = 10  # number of variables for heat map
    cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
   # f, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df_train[cols].corr().values.T, cbar=True, annot=True, square=True, yticklabels=cols.values,
                xticklabels=cols.values)
    #plt.close()
    plt.show()

    # Overall Quality vs Sale Price
    var = 'OverallQual'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
   # f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.show()

    # What! People pay more for better quality? Nothing new here. Let's move on.

    #                                -----

    # Living Area vs Sale Price
    sns.jointplot(x=df_train['GrLivArea'], y=df_train['SalePrice'], kind='reg')
    plt.show()
    # It makes sense that people would pay for the more living area.
    # What doesn't make sense is the two data points in the bottom-right of the plot.
    # We need to take care of this! What we will do is remove these outliers manually.

    # Removing outliers manually (Two points in the bottom right)
    df_train = df_train.drop(df_train[(df_train['GrLivArea'] > 4000)
                                      & (df_train['SalePrice'] < 300000)].index).reset_index(drop=True)

    # Living Area vs Sale Price
    sns.jointplot(x=df_train['GrLivArea'], y=df_train['SalePrice'], kind='reg')
    #plt.close()
    plt.show()

    # Bunları en etkili 10 parametre için yapabiliriz.

    # Up to this point, we were exploring the data
    ################################################################
