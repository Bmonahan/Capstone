from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import pandas as pd
import csv
import time,sys
from pprint import pprint
games = pd.read_csv('11_18_raw.csv',sep=',')
bruins = pd.read_csv('bruins_raw_10_11.csv',sep=',')

#print games
Bx = bruins.ix[:,0:16]
By = bruins.ix[:,16]
# Bx = games.ix[0:300,0:16]
# By = games.ix[0:300,16]
X = games.ix[:,0:16] #All stats
Y = games.ix[:,16] #Wins or losses

json_file = open('model_83pctg.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_83pctg.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.5f%%" % (loaded_model.metrics_names[1], score[1]*100))
print("%s: %.5f" % (loaded_model.metrics_names[0], score[0]))

predictions = loaded_model.predict(Bx)
# round predictions
#print predictions
rounded = [round(x[0]) for x in predictions]
#print(rounded)
print len(rounded)
import LogPlotTest.py
run()

print '--------------------------------------------------------'

with open('outcome.csv','w') as new_file:
    csv_writer = csv.writer(new_file,delimiter=',')
    header = ['wol']
    csv_writer.writerow(header)
    for i in range(0,len(rounded)):
        temp = rounded[i]
        line=[temp]
        csv_writer.writerow(line)


#print loaded_model.metrics_names[0]
