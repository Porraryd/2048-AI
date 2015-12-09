import game 
import rlagent
import sys, copy, random, time
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
### MAIN ###
TDLearner = rlagent.TDAfterstateAgent()

#TDLearner.loadWeights()

mode = 'TD'
printBoard = False


GAMES_PER_ROUND = 1000
LEARNING_GAMES = 20000
game = game.Game()
numGames = 0
scoreamount =0
maxTile = 0
win = 0
startTime = time.clock()
moves = 0
totalGames = 0
Learning = True

while True:
    if (mode == 'manual') :
        print '------------------------------'
        print 'Choose your move: (L, R, U, D)'
        print '------------------------------'
        sys.stdout.write('> ')
        line = sys.stdin.readline()
        if not line: break
        line = line.strip()
        if line == 'L' :
            game.move(LEFT)
        if line == 'R' :
            game.move(RIGHT)
        if line == 'U' :
            game.move(UP)
        if line == 'D' :
            game.move(DOWN)

    if (mode == 'random') :
        move = random.choice(tuple(game.getPossibleMoves()))
        game.move(move)
        moves += 1

    if (mode == 'TD') :
        lastState = game.getState()
        score = game.score
        move = TDLearner.chooseMove(game.getState(),game.getPossibleMoves())
        game.move(move)
        moves += 1
        score = game.score - score
        if (Learning and not game.gameOver()) :
            TDLearner.learnEval(lastState, move, score, game.getState(), game.getPreState())

        #line = sys.stdin.readline()

    if printBoard :
        game.printBoard()
        print game.getPossibleMoves()
        sys.stdin.readline()

    if (game.gameOver()) :
        #print 'Game over. Score is ' + str(game.score)

        maxTile = max(maxTile, max(game.getState()))
        numGames += 1
        if Learning : 
            totalGames += 1
            LEARNING_GAMES -= 1
        scoreamount +=game.score
        if pow(2,max(game.getState())) >= 2048:
            win += 1 

        game.reset()
        if (numGames == GAMES_PER_ROUND) :
            print 'Average score: '+ str(scoreamount/float(GAMES_PER_ROUND)) + '. Max tile : ' + str(pow(2,maxTile)) +  '. Win rate: ' + str(win/float(GAMES_PER_ROUND))
            print 'Time elapsed: ' + str(time.clock() - startTime) +'. Per game: ' + str((time.clock() - startTime)/float(GAMES_PER_ROUND))+ '. Moves/second: ' + str(moves/(time.clock() - startTime))
            print 'Total Games: ' + str(totalGames)
            numGames = 0
            win = 0
            moves = 0
            scoreamount= 0
            maxTile = 0
            if (not Learning) :
                Learning = True
                print "Learning On"
            elif (totalGames % 2000 == 0) :
                Learning = False
                print "Learning Off"
            startTime = time.clock()
            TDLearner.saveWeights()

        if LEARNING_GAMES == 0 : 
            Learning = False
