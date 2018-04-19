#From https://stackoverflow.com/questions/42432850/how-to-create-my-own-datasets-using-in-scikit-learn
#Custom dataset creation
import numpy as np
import csv
from sklearn.datasets.base import Bunch
import my_datasets

def load_my_fancy_dataset():
    with open('scikitDatasetTest.csv') as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        print 'begin'
        print temp
        print n_samples
        print n_features
        print data
        print target
        print 'end'
        for i, sample in enumerate(data_file):
            data[i] = np.asarray(sample[:-1], dtype=np.float64)
            target[i] = np.asarray(sample[-1], dtype=np.int)

    return Bunch(data=data, target=target)


