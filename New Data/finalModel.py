import csv
import time,sys
from pprint import pprint

with open('outcome.csv','w') as new_file:
    csv_writer = csv.writer(new_file,delimiter=',')
    header = ['wol']
    csv_writer.writerow(header)
    for i in range(0,16282):
        
