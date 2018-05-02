import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pydot_ng as pydot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import CSVLogger
from keras import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score,cohen_kappa_score
import os as os
import keras.callbacks as cb
from keras.datasets import mnist
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.regularizers import l1, l2
from keras.utils import np_utils
import time

seed = np.random.seed(7)
games = pd.read_csv('NHL_2011_2018.csv',sep=',')

#print games

X = games.ix[:,4:21] #All stats
y = games.ix[:,3:4] #Wins or losses

# ynum = y.values


X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
def quadDense():
    # create model
    model = Sequential()
    #model.add(Dense(32,input_dim=8,activation='relu',input_shape=(17,)))
    model.add(Dense(8,input_dim=17,activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def testModel():
    model = Sequential()
    #model.add(Dense(32,input_dim=8,activation='relu',input_shape=(17,)))
    model.add(Dense(34,input_dim=17,activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def DefineModel():
    activation_func = 'relu'
    loss_function = 'binary_crossentropy'
    #loss_function = 'mean_squared_error'

    dropout_rate = 0.4
    weight_regularizer = None
    learning_rate = 0.007

    ## Initialize model.
    model = Sequential()
    ## 1st Layer
    ## Dense' means fully-connected.
    model.add(Dense(17, input_dim=17, W_regularizer=weight_regularizer))
    model.add(Activation(activation_func))
    model.add(Dropout(0.5))

    ## 2nd Layer
    model.add(Dense(8, input_dim=17, W_regularizer=weight_regularizer))
    model.add(Activation(activation_func))
    model.add(Dropout(dropout_rate))


    ## 5th Layer
    model.add(Dense(4))
    model.add(Activation(activation_func))
    model.add(Dropout(dropout_rate))

    ## Adding Softmax Layer
    ## Last layer has the same dimension as the number of classes
    model.add(Dense(1))

    ## For classification, the activation is softmax
    model.add(Activation('softmax'))

    ## Define optimizer. we select Adam
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #opt = SGD(lr=learning_rate, clipnorm=5.)

    ## Define loss function = 'categorical_crossentropy' or 'mean_squared_error'
    model.compile(loss=loss_function, optimizer=opt, metrics=["accuracy"])
    return model


#plot_model(model, to_file='model.png')
model = DefineModel()
model.summary()

config = model.get_config()
weight = model.get_weights()

logger = CSVLogger('log.csv',separator=',',append=False)
start_time = time.time()
fit = model.fit(X_train, y_train, validation_split=0.12,epochs=50, batch_size=2, verbose=1,callbacks=[logger])


print '----------------'
print("Training took {0} seconds.".format(time.time() - start_time))
# prediction = model.predict(X_train)
# print prediction
score,acc = model.evaluate(X_test,y_test,verbose=1)
#print("%s: %.2f%%" % [model.metrics_names[1], score[2] * 100])
print 'Loss Score: ', score
print 'Accuracy: ',acc
print '--------------------------------------------------------'
os.system('python LogPlotTest.py')
