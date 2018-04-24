import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import CSVLogger
from keras import metrics
from keras.wrappers.scikit_learn import KerasClassifier
#from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score,cohen_kappa_score
import os as os

seed = np.random.seed(7)
games = pd.read_csv('NHL_2011_2018.csv',sep=',')

#print games

X = games.ix[:,4:21] #All stats
y = games.ix[:,3:4] #Wins or losses

# ynum = y.values


X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# create model
model = Sequential()
#model.add(Dense(32,input_dim=8,activation='relu',input_shape=(17,)))
model.add(Dense(24,input_dim=17,activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#plot_model(model, to_file='model.png')
model.summary()

config = model.get_config()
weight = model.get_weights()

logger = CSVLogger('log.csv',separator=',',append=False)
fit = model.fit(X_train, y_train, validation_split=0.33,epochs=25, batch_size=2, verbose=1,callbacks=[logger])


print '----------------'
prediction = model.predict(X_train)
print prediction
score,acc = model.evaluate(X_test,y_test,verbose=1)
#print("%s: %.2f%%" % [model.metrics_names[1], score[2] * 100])
print 'Loss Score: ', score
print 'Accuracy: ',acc
print '--------------------------------------------------------'
os.system('python LogPlotTest.py')
