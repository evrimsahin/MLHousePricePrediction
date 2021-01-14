import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_log_error
import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def NN(num_train, num_test):
    X = num_train.drop(['Id', 'SalePrice'], axis=1)
    y = num_train['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    num_cols = len(X_train.columns)  # get the number of columns as the number of input nodes in our network

    model = Sequential()  # initiating the model
    model.add(Dense(15, input_shape=(num_cols,), activation='relu'))  # input layer
    model.add(Dense(15, activation='relu'))  # hidden layer 1
    model.add(Dense(15, activation='relu'))  # hidden layer 2
    model.add(Dense(15, activation='relu'))  # hidden layer 3
    model.add(Dense(1, ))  # output layer

    # Compiles model
    model.compile(Adam(lr=0.003),
                  'mean_squared_error')  # optimizing method and error function, LR should be large for large outputs

    # Fits model
    history = model.fit(X_train, y_train, epochs=1000, validation_split=0.2, verbose=0)
    history_dict = history.history

    # Test how model holds up for our training data (not that good of an indicator but it gives us an approximate sense)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("The MSE score on the Train set is:\t{:0.3f}".format(np.sqrt(mean_squared_log_error(y_train, y_train_pred))))
    print("The MSE score on the Test set is:\t{:0.3f}".format(np.sqrt(mean_squared_log_error(y_test, y_test_pred))))


    num_test2 = num_test.drop("Id", axis=1)
    subm_test = model.predict(num_test2)
    subm_test_df = pd.DataFrame(subm_test, columns=['SalePrice'])

    # making a submission file
    my_submission = pd.DataFrame({'Id': num_test.Id, 'SalePrice': (subm_test_df.SalePrice)})
    # you could use any filename. We choose submission here
    my_submission.to_csv('submission_NeuralNets.csv', index=False)
