#From
#https://www.datacamp.com/community/tutorials/deep-learning-python
import pandas as pd

white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

red.head()
white.tail()
red.sample(5)
white.describe()
pd.isnull(red)

import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,2)

ax[0].hist(red.alcohol, 10 ,facecolor='red',alpha=0.5, label='Red wine')
ax[1].hist(white.alcohol,10,facecolor='white',ec='black',lw=0.5,alpha=0.5,label='White wine')

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
ax[0].legend(loc='best')
ax[1].legend(loc='best')
fig.suptitle("Distribution of Alcohol in % Vol")

#plt.show()

import numpy as np
print np.histogram(red.alcohol, bins=[7,8,9,10,11,12,13,14,15])
print np.histogram(white.alcohol, bins=[7,8,9,10,11,12,13,14,15])

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(red['quality'], red["sulphates"], color="red")
ax[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Quality")
ax[1].set_xlabel("Quality")
ax[0].set_ylabel("Sulphates")
ax[1].set_ylabel("Sulphates")
ax[0].set_xlim([0,10])
ax[1].set_xlim([0,10])
ax[0].set_ylim([0,2.5])
ax[1].set_ylim([0,2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("Wine Quality by Amount of Sulphates")

#plt.show()


np.random.seed(570)

redlabels = np.unique(red['quality'])
whitelabels = np.unique(white['quality'])

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
redcolors = np.random.rand(6,4)
whitecolors = np.append(redcolors, np.random.rand(1,4), axis=0)

for i in range(len(redcolors)):
    redy = red['alcohol'][red.quality == redlabels[i]]
    redx = red['volatile acidity'][red.quality == redlabels[i]]
    ax[0].scatter(redx, redy, c=redcolors[i])
for i in range(len(whitecolors)):
    whitey = white['alcohol'][white.quality == whitelabels[i]]
    whitex = white['volatile acidity'][white.quality == whitelabels[i]]
    ax[1].scatter(whitex, whitey, c=whitecolors[i])
    
ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlim([0,1.7])
ax[1].set_xlim([0,1.7])
ax[0].set_ylim([5,15.5])
ax[1].set_ylim([5,15.5])
ax[0].set_xlabel("Volatile Acidity")
ax[0].set_ylabel("Alcohol")
ax[1].set_xlabel("Volatile Acidity")
ax[1].set_ylabel("Alcohol") 
ax[0].legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))
ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))
fig.suptitle("Alcohol - Volatile Acidity")
fig.subplots_adjust(top=0.85, wspace=0.7)

#plt.show()

red['type'] = 1

white['type'] = 0

wines = red.append(white,ignore_index=True)

from sklearn.model_selection import train_test_split

X = wines.ix[:,0:11]

y = np.ravel(wines.type)
print 'RAVEL'
print y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(11,)))
model.add(Dense(8,activation ='relu'))
model.add(Dense(1,activation='sigmoid'))

model.output_shape
model.summary()
model.get_config()
model.get_weights()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=20,batch_size=1,verbose=1)

y_pred = model.predict(X_test)

y_pred[:5]
y_test[:5]

score = model.evaluate(X_test,y_test,verbose=1)
print score

from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score,cohen_kappa_score

confusion_matrix(y_test,y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)
cohen_kappa_score(y_test,y_pred)

y = wines.quality
X = wines.drop('quality',axis=1)

X=StandardScalar().fit_transform(X)

model = Sequential()
model.add(Dense(64,input_dim=12,activation='relu'))
model.add(Dense(1))

from sklearn.model_selection import StratifiedKFold

seed = 7
np.random.seed(seed)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(64, input_dim=12, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.fit(X[train], Y[train], epochs=10, verbose=1)

mse_value, mae_value = model.evaluate(X[test], Y[test], verbose=0)
print mse_value

print mae_value

from sklearn.metrics import r2_score

r2_score(Y[test], y_pred)
