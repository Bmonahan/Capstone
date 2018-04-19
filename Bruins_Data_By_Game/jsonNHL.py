import json
from pprint import pprint

with open('nhl.json') as data_file:
    data = json.load(data_file)
game = 1
total = 1
try:
    for i in range(409,-1,-1):
        #pprint(data["data"][i]["gameDate"])
        date = data['data'][i]['gameDate']
        if game==83:
            game = 1
        print "Game: "+str(game)+" Game Total: "+str(total)
        print data
        game+=1
        total +=1
except IndexError:
    print""

##pprint(data["data"])
