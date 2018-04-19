import json
import csv
from pprint import pprint
#16281 MAX
maxData = 16282
data = json.load(open('2011_12-2017-18.json'))

with open('NHL_2011_2018.csv','w') as new_file:
    csv_writer = csv.writer(new_file,delimiter=',')
    header = ['gameID','tm','oppTm','loc','ga','gf','fowP','foL','foW','pkPctg','ppPctg','sf','sa']
    csv_writer.writerow(header)
    for i in range(0,maxData):
        gameId = str(data["data"][i]["gameId"])
        #print(gameId)
        teamAbb = str(data["data"][i]["teamAbbrev"])
        oppTm = str(data['data'][i]['opponentTeamAbbrev'])
        #print(teamAbb)
        gameLoc = str(data["data"][i]["gameLocationCode"])
        
        goalAgainst = str(data["data"][i]["goalsAgainst"])
        
        goalFor = str(data["data"][i]["goalsFor"])
        
        foWP = str(data["data"][i]["faceoffWinPctg"])
        
        foL = str(data["data"][i]["faceoffsLost"])
        
        foW = str(data["data"][i]["faceoffsWon"])
        
        pkPctg = str(data["data"][i]["penaltyKillPctg"])
        
        ppPctg = str(data["data"][i]["ppPctg"])
        
        sf = str(data["data"][i]["shotsFor"])
        
        sa = str(data["data"][i]["shotsAgainst"])
        line = [gameId,teamAbb,oppTm,gameLoc,goalAgainst,goalFor,foWP,foL,foW,pkPctg,ppPctg,sf,sa]
        print line
        #line = gameId+','+teamAbb+','+gameLoc+','+goalFor+','+goalAgainst+','+foWP+','+foL+','+foW+','+pkPctg+','+ppPctg+','+sf+','+sa
        csv_writer.writerow(line)

        #print(data["data"][i]["teamAbbrev"])
