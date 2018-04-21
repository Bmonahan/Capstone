import json
import csv
from pprint import pprint

maxData = 16282
data = json.load(open('SAT_11_18.json'))

with open('SAT_2011_2018.csv','w') as new_file:
    csv_writer = csv.writer(new_file,delimiter=',')
    header = ['GameID','teamAbbrev','SATP','OZF','DZF','SPSv']
    csv_writer.writerow(header)
    for i in range(0,maxData):
        gameId = str(data["data"][i]["gameId"])
        teamAbb = str(data["data"][i]["teamAbbrev"])
        SATP = str(data['data'][i]['shotAttemptsPctg'])
        OZF = str(data['data'][i]['offensiveZoneFaceoffs'])
        DZF = str(data['data'][i]['defensiveZoneFaceoffs'])
        SPSv = str(data['data'][i]['shootingPlusSavePctg'])

        line = [gameId,teamAbb,SATP,OZF,DZF,SPSv]
        print line
        csv_writer.writerow(line)
