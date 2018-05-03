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
games = pd.read_csv('11_18_raw.csv',sep=',')

#print games

X = games.ix[:,0:16] #All stats
y = games.ix[:,16] #Wins or losses

# ynum = y.values


X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.27, random_state=seed, shuffle=True)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
def quadDense():
    # create model
    model = Sequential()
    model.add(Dense(16,input_shape=(16,), kernel_initializer='normal'))
    model.add(Activation('relu'))
    model.add(Dense(8, kernel_initializer='normal'))
    model.add(Activation('relu'))
    model.add(Dense(4, kernel_initializer='normal'))
    model.add(Activation('relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.add(Activation('sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#plot_model(model, to_file='model.png')
model = quadDense()
model.summary()

config = model.get_config()
weight = model.get_weights()

logger = CSVLogger('log.csv',separator=',',append=False)
start_time = time.time()
fit = model.fit(X_train, y_train, validation_split=0.33,epochs=25, batch_size=4, verbose=1,callbacks=[logger])


print '----------------'
print("Training took {0} seconds.".format(time.time() - start_time))

score,acc = model.evaluate(X_test,y_test,verbose=1)
#print("%s: %.2f%%" % [model.metrics_names[1], score[2] * 100])
print 'Loss Score: ', score
print 'Accuracy: ',acc
print '--------------------------------------------------------'
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
#print(rounded)
print len(rounded)

print '--------------------------------------------------------'

model_json = model.to_json()
with open("model_83pctg.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_83pctg.h5")
print("Saved model to disk")

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
#
# loaded_model.load_weights("model.h5")
# print("Loaded")
#
# loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
