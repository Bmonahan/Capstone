import bruins_dataset
import numpy as np
import pandas as pd

from pandas import Series, DataFrame

import scipy
from scipy.stats import spearmanr

from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

import sklearn
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
ds = bruins_dataset.load_set()


model = LogisticRegression()
model.fit(ds.data,ds.target)

#print model

X = ds.data
Y = ds.target

#expected = ds.target
#predicted = model.predict(ds.data)

pred = model.predict(X)
print pred

print(metrics.classification_report(Y,pred))
#print(metrics.confusion_matrix(Y,pred))

print 'XXXXXXX'
#print predicted
from sklearn.metrics import classification_report

print(classification_report(ds.data, pred))



#X = ds.data
#Y = ds.target
#print X
#print Y
