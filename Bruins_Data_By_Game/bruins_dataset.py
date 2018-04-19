#From https://stackoverflow.com/questions/42432850/how-to-create-my-own-datasets-using-in-scikit-learn
#Formatted to work with my data

import numpy as num
import csv
from sklearn.datasets.base import Bunch

def load_set():
    with open('Bruins_2005_2007.csv') as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        print temp
        n_samples = 164
        n_features = 9
        keys = num.empty((temp,))
        data = num.empty((n_samples, n_features))
        target = num.empty((n_samples,), dtype=num.int)
        key = num.asarray((temp,), dtype=list)
        for i, sample in enumerate(data_file):
            
            data[i] = num.asarray(sample[:-2], dtype=num.int)
            target[i] = num.asarray(sample[-2], dtype=num.int)
    return Bunch(data=data, target=target)

load_set()
