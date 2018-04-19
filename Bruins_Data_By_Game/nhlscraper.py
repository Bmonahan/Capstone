from nhlscrapi.games.game import Game, GameKey, GameType
from nhlscrapi.games.cumstats import Score, ShotCt, Corsi, Fenwick
import io, json
for i in range(1,1231):
    season = 2011                                    # 2010-2011 season
    game_num =  i
    gp = 1#
    game_type = GameType.Regular
    try:# regular season game
        game_key = GameKey(season, game_type, game_num)
        print "Game Number: ",i
        # define stat types that will be counted as the plays are parsed
        cum_stats = {
          'Score': Score(),
          'Shots': ShotCt(),
          'Corsi': Corsi(),
          'Fenwick': Fenwick()
        }
        game = Game(game_key, cum_stats=cum_stats)

        ##print('\nRefs          : {}'.format(game.refs))
        ##print('Linesman      : {}'.format(game.linesman))
        # print('Coaches')
        # print('  Home        : {}'.format(game.home_coach))
        # print('  Away        : {}'.format(game.away_coach))
        # print ""
        teams =  str(game.cum_stats['Score'].total)
        t1 = teams[2]+teams[3]+teams[4]

        t2 = teams[12]+teams[13]+teams[14]

        ##add game data to a text file just for Bruins Games in the 2011 Season
        if t1 == 'BOS' or t2 == 'BOS':
        #print('Final         : {}'.format(game.cum_stats['Score'].total))
        #print('Shootout      : {}'.format(game.cum_stats['Score'].shootout.total))
        #print('Shots         : {}'.format(game.cum_stats['Shots'].total))
            print"GOT A GAME"
            with open("scrapperOut.txt","a") as text_file:
                text_file.write("2011 Game: %s\n" % gp)
                text_file.write('Final         : {}\n'.format(game.cum_stats['Score'].total))
                text_file.write('EV Shot Atts  : {}\n'.format(game.cum_stats['Corsi'].total))
                text_file.write('Corsi         : {}\n'.format(game.cum_stats['Corsi'].share()))
                text_file.write('FW Shot Atts  : {}\n'.format(game.cum_stats['Fenwick'].total))
                text_file.write('Fenwick       : {}\n'.format(game.cum_stats['Fenwick'].share()))
                text_file.write("-----------------------------------------------------------\n")
            gp += 1


        # scrape all remaining reports
        game.load_all()
    except KeyError:
        print"Game Doesn't exist"
