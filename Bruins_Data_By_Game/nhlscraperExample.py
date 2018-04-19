from nhlscrapi.games.game import Game, GameKey, GameType
from nhlscrapi.games.cumstats import Score, ShotCt, Corsi, Fenwick
import io, json
for i in range(10,13):
    season = 2011                                    # 2010-2011 season
    game_num =  i
    gp = 1
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
        print('Final         : {}'.format(game.cum_stats['Score'].total))
        print('Shootout      : {}'.format(game.cum_stats['Score'].shootout.total))
        print('Shots         : {}'.format(game.cum_stats['Shots'].total))
        print('EV Shot Atts  : {}'.format(game.cum_stats['Corsi'].total))
        print('Corsi         : {}'.format(game.cum_stats['Corsi'].share()))
        print('FW Shot Atts  : {}'.format(game.cum_stats['Fenwick'].total))
        print('Fenwick       : {}'.format(game.cum_stats['Fenwick'].share()))


        game.load_all()
    except KeyError:
        print"Game Doesn't exist"