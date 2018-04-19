import json
import csv

bruins = json.load(open('Bruins_05_07_FO.json'))

with open('Bruins_2005_2007 copy.csv','r') as csv_file:
    csv_reader = csv.reader(csv_file)
    print 'Reading...'
    with open('Bruins_2005_2007.csv','w') as new_file:
        csv_writer = csv.writer(new_file,delimiter=',')
        print 'Writing...'
        i = 0
        for line in csv_reader:
            line[9]=bruins['data'][i]['faceoffWinPctg']
            line[10] = bruins['data'][i]['blockedShots']
            line[11] = bruins['data'][i]['giveaways']
            line[12] = bruins['data'][i]['shootingPctg']
            line[13] = bruins['data'][i]['takeaways']
            line[14] = bruins['data'][i]['hits']
            csv_writer.writerow(line)
            print line
            i+= 1

print 'Done...'