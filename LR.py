import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


def LR(num_train, num_test, test_data):
    X = num_train.drop(['Id', 'SalePrice'], axis=1)
    y = num_train['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr.score(X_test, y_test)
    print(lr.score(X_test, y_test))
    pred = lr.predict(num_test.drop(labels=['Id'], axis=1))
    pred2 = lr.predict(X_test)
    Submission = pd.DataFrame(data=pred, columns=['SalePrice'])
    Submission.head()
    Submission['Id'] = test_data['Id']
    Submission.set_index('Id', inplace=True)
    Submission.head()
    Submission.to_csv('Submission.csv')
    print("The MSE score on the Train set is:\t{:0.3f}".format(np.sqrt(mean_squared_log_error(y_test, pred2))))
