import json
import csv
import time,sys
from pprint import pprint

#https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
#16281 MAX
maxData = 16282
data = json.load(open('2011_12-2017-18.json'))
sat = json.load(open('SAT_11_18.json'))
hit = json.load(open('HIT_11_18.json'))
with open('NHL_2011_2018.csv','w') as new_file:
    csv_writer = csv.writer(new_file,delimiter=',')
    header = ['gameID','tm','oppTm','wol','loc','ga','gf','fowP','foL','foW','pkPctg','ppPctg','sf','sa','satp','ozf','dzf','spsv','hit','bks','tka','gva']
    csv_writer.writerow(header)
    for i in range(0,maxData):
        # percent = str((float(i)/maxData)*100)
        # print percent[0:2]+'%'
        index = 0
        team = ''
        gameId = str(data["data"][i]["gameId"])
        #SAT DATA
        teamAbb = str(data["data"][i]["teamAbbrev"])
        for x in range(0,maxData):
            satGID = str(sat['data'][x]['gameId'])
            satTm = str(sat['data'][x]['teamAbbrev'])
            if satGID == gameId and satTm == teamAbb:
                #print x
                SATP = str(sat['data'][x]['shotAttemptsPctg']*100)
                OZF = str(sat['data'][x]['offensiveZoneFaceoffs'])
                DZF = str(sat['data'][x]['defensiveZoneFaceoffs'])
                SPSv = str(sat['data'][x]['shootingPlusSavePctg']*100)
                break
        for y in range(0,maxData):
            hitGID = str(hit['data'][y]['gameId'])
            hitTm = str(hit['data'][y]['teamAbbrev'])
            if hitGID == gameId and hitTm == teamAbb:
                index = y
                hits = str(hit['data'][y]['hits'])
                bks = str(hit['data'][y]['blockedShots'])
                tka = str(hit['data'][y]['takeaways'])
                gva = str(hit['data'][y]['giveaways'])
                break

        # sDat.write(teamAbb+"\n")
        # sDat.write(gameId+"\n")
        oppTm = str(data['data'][i]['opponentTeamAbbrev'])
        #print(teamAbb)
        wins = str(data['data'][i]['wins'])
        #loss = str(data['data'][i]['losses'])
        if wins =='1':
            wol = '1'
        else:
            wol = '0'

        gameLoc = str(data["data"][i]["gameLocationCode"])
        if gameLoc == 'H':
            gameLoc = '1'
        else:
            gameLoc = '0'

        goalAgainst = str(data["data"][i]["goalsAgainst"])

        goalFor = str(data["data"][i]["goalsFor"])

        foWP = str(data["data"][i]["faceoffWinPctg"]*100)

        foL = str(data["data"][i]["faceoffsLost"])

        foW = str(data["data"][i]["faceoffsWon"])

        pkPctg = str(data["data"][i]["penaltyKillPctg"]*100)

        ppPctg = str(data["data"][i]["ppPctg"]*100)

        sf = str(data["data"][i]["shotsFor"])

        sa = str(data["data"][i]["shotsAgainst"])

        line = [gameId,teamAbb,oppTm,wol,gameLoc,goalAgainst,goalFor,foWP,foL,foW,pkPctg,ppPctg,sf,sa,SATP,OZF,DZF,SPSv,hits,bks,tka,gva]
        progress(i, maxData, status=gameId+' '+teamAbb+' '+oppTm)
        #print gameId+' '+teamAbb+' '+oppTm

        #line = gameId+','+teamAbb+','+gameLoc+','+goalFor+','+goalAgainst+','+foWP+','+foL+','+foW+','+pkPctg+','+ppPctg+','+sf+','+sa
        csv_writer.writerow(line)

        #print(data["data"][i]["teamAbbrev"])
