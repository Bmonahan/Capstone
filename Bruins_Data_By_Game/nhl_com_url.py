import requests
import lxml.html
from pprint import pprint
from sys import exit
import json
import csv

url = 'http://www.nhl.com/stats/rest/grouped/team/basic/game/teamsummary?cayenneExp=gameDate%3E=%222005-10-05T04:00:00.000Z%22%20and%20gameDate%3C=%222010-04-12T03:59:59.999Z%22%20and%20gameTypeId=%222%22%20and%20teamId=6&factCayenneExp=gamesPlayed%3E=1&sort=[{%22property%22:%22gameId%22,%22direction%22:%22DESC%22}]'
resp = requests.get(url).text
print resp
resp = json.loads(resp)

pprint(resp['data'])
