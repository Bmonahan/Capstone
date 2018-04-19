#From https://stackoverflow.com/questions/42432850/how-to-create-my-own-datasets-using-in-scikit-learn
#Custom dataset loading

import my_datasets
mfd = my_datasets.load_my_fancy_dataset()
X = mfd.data
Y = mfd.target

print X
print Y
