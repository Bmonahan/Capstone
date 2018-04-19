import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model,datasets
from pprint import pprint

df = pd.read_csv('Bruins_2005_2010.csv')

target = df['outcome']
data = df.drop('outcome',axis=1)

print data
h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(data,target)
#pprint(df)