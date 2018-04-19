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

iris = datasets.load_iris()
model = LogisticRegression()
model.fit(iris.data,iris.target)
print model
expected = iris.target
predicted = model.predict(iris.data)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

sb.set(style='darkgrid')

#Change for new dataset format
seasons = pd.read_csv('Bruins_2005_2007.csv')
seasons.columns = ['team','ha','wl','sf','pimf','ppgf','shgf','sa','pima','ppga','shga']
seasons.head()
print seasons.head()
#print seasons.target()

season_data = seasons.ix[:,(3,9)].values
season_data_names = ['sf','sa']

y = seasons.ix[:,2].values
print 'Y'
print y

#print y
#print len(y)

a = sb.regplot(x='sf',y='sa',data=seasons,scatter= True)

sf = seasons['sf']
sa = seasons['sa']

spearmanr_coefficient, p_value = spearmanr(sf,sa)
print 'Rank coeff %0.3f' % (spearmanr_coefficient)
print seasons.isnull().sum()

#x = sb.countplot(x='wl',data=seasons,palette='hls')

print seasons.info()

X = scale(season_data)
LogReg = LogisticRegression()

LogReg.fit(X,y)
print LogReg.score(X,y)

y_pred = LogReg.predict(X)
from sklearn.metrics import classification_report

print(classification_report(y, y_pred))
