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
from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score,cohen_kappa_score

bruins = pd.read_csv('Bruins_2005_2007.csv',sep=',')

X = bruins.ix[:, 0:13]
y = bruins.ix[:, 13:14]

y_numpy = y.values
y_list = y_numpy.tolist()
seed = 7
np.random.seed(seed)

X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.43, random_state=42)

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12,activation='relu',input_shape=(13,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_bruinsfull_model():
    model = Sequential()
    model.add(Dense(32,input_dim=13,activation='relu'))
    #model.add(Flatten())
    model.add(Dense(12,activation='relu'))
    model.add(Dense(1,activation='relu'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
model.summary()

config = model.get_config()
weight = model.get_weights()

logger = CSVLogger('log.csv',separator=',',append=False)
fit = model.fit(X_train, y_train, validation_split=0.33,epochs=150, batch_size=2, verbose=1,callbacks=[logger])
y_test_pred = model.predict(X_train)


print '------- Comparing Prediction to Acutal---------'
y_pred_list = y_test_pred.tolist()
for pred,lis in zip(y_pred_list,y_list):
    wl = str(lis)
    p = str(pred).strip()
    p = p[1:7]
    print '%s vs. %.7f'  % (wl,float(p))

print '----------------'

score,acc = model.evaluate(X_test,y_test,verbose=1)
#print("%s: %.2f%%" % [model.metrics_names[1], score[2] * 100])
print 'Loss Score: ', score
print 'Accuracy: ',acc*100
print '--------------------------------------------------------'
# bruins_model = create_bruinsfull_model()
# model.summary()
# bruins_fit = bruins_model.fit(X, y, validation_split=.33, epochs=200,batch_size=10, verbose=1)
#
# y_pred = bruins_model.predict(X,batch_size=10,verbose=1)
# score, acc = bruins_model.evaluate(X, y, verbose=1)
# print 'Score: ', score
# print 'Accuracy: ', acc*100




