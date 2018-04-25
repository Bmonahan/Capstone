import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.ticker import LinearLocator

def run():
    data = pd.read_csv('log.csv',sep=',')

    x = data.ix[:,0:1]
    valAcc = data.ix[:,1:2]
    valLoss = data.ix[:,2:3]
    print valAcc
    plt.plot(x,valAcc)
    plt.plot(x,valLoss)
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    plt.ylabel('Accuracy/Loss')
    plt.xlabel('Epochs')
    #plt.get_yaxis().set_major_locator(LinearLocator(numticks=100))
    plt.show()
    exit()

run()

def sfvssa():
    data = pd.read_csv('NHL_2011_2018.csv',sep=',')

    sf = data.ix[:,12]
    sa = data.ix[:,13]
    plt.plot(sf,sa)
    plt.ylim([0,70])
    plt.xlim([0,70])
    plt.ylabel('sa')
    plt.xlabel('sf')
    plt.show()
