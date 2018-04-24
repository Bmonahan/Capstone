import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run():
    data = pd.read_csv('log.csv',sep=',')

    x = data.ix[:,0:1]
    valAcc = data.ix[:,1:2]
    valLoss = data.ix[:,2:3]
    plt.plot(x,valAcc)
    plt.plot(x,valLoss)
    plt.ylabel('Accuracy/Loss')
    plt.xlabel('Epochs')
    plt.show()

run()
