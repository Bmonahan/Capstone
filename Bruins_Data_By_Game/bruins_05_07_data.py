import numpy as num
import csv
from sklearn.datasets.base import Bunch

def load05_07_set():
    with open('Bruins_2005_2007.csv') as csv_file:
        dataFile = csv.reader(csv_file)
        temp = next(dataFile)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples,n_features))
        target = np.empty((n_samples,),dtype=np.int)

        for i, sample in enumerate(data_file):
            data[i] = np.asarray(sample[:-1],dtype=np.float64)
            target[i] = np.asarray(sample[-1],dtype=np.int)


    return Bunch(data=data, target=target)
