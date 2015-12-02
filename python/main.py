import game 
import rlagent
import sys, copy, random, time

### MAIN ###
TDLearner = rlagent.TDAgent()

TDLearner.loadWeights()

mode = 'TD'
printBoard = False

game = game.Game()
numGames = 0
scoreamount =0
maxTile = 0
win = 0
startTime = time.clock()
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
        move = random.randint(0,3)
        game.move(move)

    if (mode == 'TD') :
        lastState = game.getState()
        score = game.score
        move = TDLearner.chooseMove(game.getState())
        game.move(move)
        score = game.score - score
        TDLearner.learnEval(lastState, move, score, game.getState(), game.getPreState())

        #line = sys.stdin.readline()

    if printBoard :
        game.printBoard()
        sys.stdin.readline()

    if (game.gameOver()) :
        #print 'Game over. Score is ' + str(game.score)

        maxTile = max(maxTile, max(game.getState()))
        numGames += 1
        scoreamount +=game.score
        if pow(2,max(game.getState())) >= 2048:
            win += 1 

        game.reset()
        if (numGames == 20) :
            print 'Average score: '+ str(scoreamount/float(20)) + '. Max tile : ' + str(pow(2,maxTile)) +  '. Win rate: ' + str(win/float(20))
            print 'Time elapsed: ' + str(time.clock() - startTime) +'. Per game: ' + str((time.clock() - startTime)/float(20))
            numGames = 0
            win = 0
            scoreamount= 0
            maxTile = 0
            startTime = time.clock()
            TDLearner.saveWeights()

