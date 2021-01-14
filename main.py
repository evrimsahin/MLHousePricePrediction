import pandas as pd
import data_visualization
import data_processing
import LR
import neuralnetwork
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#data_visualization.visualization(df_train)

num_train, num_test = data_processing.processing(df_train,df_test)

neuralnetwork.NN(num_train,num_test)

