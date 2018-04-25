import keras.callbacks as cb
from keras.datasets import mnist
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.regularizers import l1, l2
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
import time
import pandas as pd
games = pd.read_csv('NHL_2011_2018.csv',sep=',')
# header = ['gameID','tm','oppTm','wol','loc','ga','gf','fowP','foL','foW','pkPctg','ppPctg','sf','sa','satp','ozf','dzf','spsv','hit','bks','tka','gva']

def PreprocessDataset():
    from sklearn import preprocessing
    data = pd.read_csv('NHL_2011_2018.csv',index_col=False,low_memory=False)
    data = data.reindex(np.random.permutation(data.index))


    # x = data[x_columns]
    # y = data['wol']
    x = data.ix[:,4:21] #All stats
    y = data.ix[:,3:4] #Wins or losses

    train_max_row = int(data.shape[0]*0.9)

    x_train = x.iloc[:train_max_row]
    x_test = x.iloc[train_max_row:]

    y_train = y.iloc[:train_max_row]
    y_test = y.iloc[train_max_row:]

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    ################Pre-processing###########
    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = PreprocessDataset()

def DefineModel():
    activation_func = 'relu'
    loss_function = 'categorical_crossentropy'
    #loss_function = 'mean_squared_error'

    dropout_rate = 0.4
    weight_regularizer = None
    learning_rate = 0.005

    ## Initialize model.
    model = Sequential()
    ## 1st Layer
    ## Dense' means fully-connected.
    model.add(Dense(128, input_dim=17, W_regularizer=weight_regularizer))
    model.add(Activation(activation_func))
    model.add(Dropout(0.5))

    ## 2nd Layer
    model.add(Dense(64, input_dim=128, W_regularizer=weight_regularizer))
    model.add(Activation(activation_func))
    model.add(Dropout(dropout_rate))

    ## 3rd Layer
    model.add(Dense(32))
    model.add(Activation(activation_func))
    model.add(Dropout(dropout_rate))

    ## 4th Layer
    model.add(Dense(16))
    model.add(Activation(activation_func))
    model.add(Dropout(dropout_rate))

    ## 5th Layer
    model.add(Dense(8))
    model.add(Activation(activation_func))
    model.add(Dropout(dropout_rate))

    ## Adding Softmax Layer
    ## Last layer has the same dimension as the number of classes
    model.add(Dense(3))

    ## For classification, the activation is softmax
    model.add(Activation('softmax'))

    ## Define optimizer. we select Adam
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #opt = SGD(lr=learning_rate, clipnorm=5.)

    ## Define loss function = 'categorical_crossentropy' or 'mean_squared_error'
    model.compile(loss=loss_function, optimizer=opt, metrics=["accuracy"])
    return model

def TrainModel(data=None, epochs=20):
    batch=128
    start_time = time.time()
    model = DefineModel()
    if data is None:
        print("Must provide data.")
        return
    x_train, x_test, y_train, y_test = data
    print('Start training.')
    ## Use the first 55,000 (out of 60,000) samples to train, last 5,500 samples to validate.
    history = model.fit(x_train, y_train, validation_split=0.33,epochs=25, batch_size=3, verbose=1)
    print("Training took {0} minutes.".format((time.time() - start_time))/60)
    return model, history

trained_model, training_history = TrainModel(data=[x_train, x_test, y_train, y_test],epochs = 50)

def TestModel(model=None, data=None):
    if model is None:
        print("Must provide a trained model.")
        return
    if data is None:
        print("Must provide data.")
        return
    x_test, y_test = data
    scores = model.evaluate(x_test, y_test)
    return scores

test_score = TestModel(model=trained_model, data=[x_test, y_test])
print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100))
