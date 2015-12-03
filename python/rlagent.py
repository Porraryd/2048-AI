import collections
import game
import copy, random
import pickle
import numpy
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

LEARNING_RATE = 0.01


class TDAgent() :
    def __init__(self) :
        self.weights = collections.Counter()

    #Chooses the most beneficial move and returns it
    #If tie, choose at random
    def chooseMove(self, state, possibleMoves) : 

        #If there is only one move, just return it
        if len(possibleMoves) == 1 :
            return tuple(possibleMoves)[0]

        #Calculate possible moves and their afterstate
        moves = []

        simpleTD = False

        for i in possibleMoves :
            newGame = game.Game(state)
            newGame.move(i)
            newState = newGame.getPreState()
            if newState == state :
                continue

            if simpleTD :
                moves.append((newGame.score + self.V(newState), i))
                continue
            #Calculate the possible transitions and their probabilities
            possibleNextStates = []
            possibleZeros = [index for index, e in enumerate(newState) if e == 0]
            twoProb = 1/float(len(possibleZeros))*0.9
            fourProb = 1/float(len(possibleZeros))*0.1
            for j in possibleZeros : 
                possibleState = list(newState)
                possibleState[j] = 1
                possibleNextStates.append((tuple(possibleState), twoProb))
                possibleState[j] = 2
                possibleNextStates.append((tuple(possibleState), fourProb))

            #Create the final V
            sumV = 0
            for posState in possibleNextStates :
                sumV += posState[1] * self.V(posState[0])

            #Add the move to the list of moves
            moves.append((newGame.score + sumV, i))

        ##Return randomly between the best moves
        maxR = max(moves)[0]
        bestMoves = [a for (r,a) in moves if isclose(r, maxR)]

        return random.choice(bestMoves)
        return max(moves)[1]


    #Takes a state, action, reward and learns from it using TD-learning
    def learnEval(self, state, action, reward, newState, afterState) :
        newV = self.V(newState)
        V = self.V(state)

        for f, v in self.featureExtractor(state).iteritems():
            self.weights[f] = self.weights[f] - (LEARNING_RATE*(V-(reward+(1*newV)))*v)


    def saveWeights(self):
        # print 'self. s',self.weights
        with open('saved_weightsTDtest.txt', 'w') as file_:
            pickle.dump(self.weights, file_)

    def loadWeights(self):
        with open('saved_weightsTDtest.txt','r') as myFile:
            self.weights = pickle.load(myFile)
            print "Loaded: %d weights" % len(self.weights)


    #Calculate the estimated value of a state using the current weight vector
    def V(self, state) :
        features = self.featureExtractor(state)
        V = sum(self.weights.get(f, 0) * v for f, v in features.items())
        return V

    #Features extractor splitting the game board into smaller tiles
    def featureExtractor(self, state) :
        features = collections.Counter()
        #rows
        features['rows1', state[0:4]] = 1
        features['rows2', state[4:8]] = 1
        features['rows3', state[8:12]] = 1
        features['rows4', state[12:16]] = 1

        #columns
        features['col1', state[0:16:4]] = 1
        features['col2', state[1:16:4]] = 1
        features['col3', state[2:16:4]] = 1
        features['col4', state[3:17:4]] = 1

        #2x2 grids
        features['2x21', state[0:2]+state[4:6]] = 1
        features['2x22', state[1:3]+state[5:7]] = 1
        features['2x23', state[2:4]+state[6:8]] = 1
        features['2x24', state[4:6]+state[8:10]] = 1
        features['2x25', state[5:7]+state[9:11]] = 1
        features['2x26', state[6:8]+state[10:12]] = 1
        features['2x27', state[8:10]+state[12:14]] = 1
        features['2x28', state[9:11]+state[13:15]] = 1
        features['2x29', state[10:12]+state[14:16]] = 1

        return features


    def symmetryFeatureExtractor(self, state) :
        features = collections.Counter()
        #rows
        features['rowOuter', state[0:4]] += 1
        features['rowOuter', state[::-1][12:16]] += 1
        features['rowOuter', state[12:16]] += 1
        features['rowOuter', state[::-1][0:4]] += 1

        features['rowOuter', state[0:16:4]] += 1
        features['rowOuter', state[::-1][3:17:4]] += 1
        features['rowOuter', state[3:17:4]] += 1
        features['rowOuter', state[::-1][0:16:4]] += 1

        features['rowInner', state[4:8]] += 1
        features['rowInner', state[::-1][8:12]] += 1
        features['rowInner', state[8:12]] += 1
        features['rowInner', state[::-1][4:8]] += 1

        features['rowInner', state[1:16:4]] += 1
        features['rowInner', state[2:16:4]] += 1
        features['rowInner', state[::-1][1:16:4]] += 1
        features['rowInner', state[::-1][2:16:4]] += 1
        '''
        #3x2 grids  
       
        features['3x2Outer', state[0:2]+state[4:6] + state[8:10]] += 1
        features['3x2Outer', state[::-1][0:2]+state[::-1][4:6] + state[::-1][8:10]] += 1

        features['3x2Outer', state[4:6]+state[8:10]+state[12:14]] += 1
        features['3x2Outer', state[::-1][4:6]+state[::-1][8:10]+state[::-1][12:14]] += 1

        features['3x2Outer', state[2:4]+state[6:8] + state[10:12]] += 1
        features['3x2Outer', state[::-1][2:4]+state[::-1][6:8] + state[::-1][10:12]] += 1

        features['3x2Outer', state[6:8]+state[10:12]+state[14:16]] += 1
        features['3x2Outer', state[::-1][6:8]+state[::-1][10:10]+state[::-1][12:14]] += 1

        features['2x2corner', state[0]+state[1]+state[4]+state[5]] +=1
        features['2x2corner', state[0]+state[4]+state[1]+state[5]] +=1

        features['2x2corner', state[3]+state[2]+state[7]+state[6]] +=1
        features['2x2corner', state[3]+state[7]+state[2]+state[6]] +=1

        features['2x2corner', state[12]+state[13]+state[8]+state[9]] +=1
        features['2x2corner', state[12]+state[8]+state[13]+state[9]] +=1

        features['2x2corner', state[15]+state[14]+state[11]+state[10]] +=1
        features['2x2corner', state[15]+state[11]+state[14]+state[10]] +=1

        features['2x2inner', state[1]+state[2]+state[5]+state[6]] +=1
        features['2x2inner', state[2]+state[1]+state[6]+state[5]] +=1

        features['2x2inner', state[4]+state[8]+state[5]+state[9]] +=1
        features['2x2inner', state[8]+state[4]+state[9]+state[5]] +=1

        features['2x2inner', state[13]+state[14]+state[9]+state[10]] +=1
        features['2x2inner', state[14]+state[13]+state[10]+state[9]] +=1

        features['2x2inner', state[7]+state[11]+state[6]+state[10]] +=1
        features['2x2inner', state[11]+state[7]+state[10]+state[6]] +=1
        '''

        return features
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class TDAfterstateAgent() :
    def __init__(self) :
        self.weights = collections.Counter()

    #Chooses the most beneficial move and returns it
    #If tie, choose at random
    def chooseMove(self, state, possibleMoves) : 
        #Calculate possible moves and their afterstates
                #If there is only one move, just return it
        if len(possibleMoves) == 1 :
            return tuple(possibleMoves)[0]

        #Calculate possible moves and their afterstate
        moves = []
        for i in possibleMoves :
            moves.append((self.evaluate(state, i),i))

        ##Return randomly between the best moves
        maxR = max(moves)[0]
        bestMoves = [a for (r,a) in moves if isclose(r, maxR)]

        return random.choice(bestMoves)
        return max(moves)[1]

    #Compute all afterstates possible. Returns a list with (reward, action, afterstate) tuples.
    def computeAfterStates(self, state) :
        moves = []
        for i in range(4) :
            newGame = game.Game(state)
            newGame.move(i)
            newState = newGame.getPreState()
            if newState != state : 
                moves.append((newGame.score, i, newState))

        return moves
    
    def computeAfterstate(self, state, action) :
            newGame = game.Game(state)
            newGame.move(action)
            return (newGame.score, newGame.getPreState())

    def evaluate(self, state, action) :
            (reward, afterstate) = self.computeAfterstate(state, action)
            return reward + self.V(afterstate)

    #Takes a state, action, reward and learns from it using TD-learning
    def learnEval(self, state, action, reward, newState, afterState) :
        moves = []
        newGame = game.Game(newState)
        for i in newGame.getPossibleMoves() :
            (reward, afterstate) = self.computeAfterstate(newState, i)
            moves.append((reward + self.V(afterstate), i, afterstate))
        if not moves :
            return
        aNext = max(moves)
        sNext = aNext[2]
        newV = self.V(sNext)
        rNext = aNext[0] - newV

        V = self.V(afterState)

        # print 'self.weights ', self.weights
        for f, v in self.featureExtractor(afterState).iteritems():
            self.weights[f] = self.weights[f] - (LEARNING_RATE*(V-(reward+(1*newV)))*v)


    def saveWeights(self):
        # print 'self. s',self.weights
        with open('saved_weightsAS.txt', 'w') as file_:
            print "Saving weights. Do not close me please."
            pickle.dump(self.weights, file_)
            print "Saving done."

    def loadWeights(self):
        with open('saved_weightsAS.txt','r') as myFile:
            self.weights = pickle.load(myFile)
            print "Loaded: %d weights" % len(self.weights)


    #Calculate the estimated value of a state using the current weight vector
    def V(self, state) :
        features = self.featureExtractor(state)
        return dotProduct(features, self.weights)

    #Features extractor splitting the game board into smaller tiles
    def featureExtractor(self, state) :
        features = collections.Counter()
        #rows
        features['rows1', state[0:4]] = 1
        features['rows2', state[4:8]] = 1
        features['rows3', state[8:12]] = 1
        features['rows4', state[12:16]] = 1
        features['col1', state[0:16:4]] = 1
        features['col2', state[1:16:4]] = 1
        features['col3', state[2:16:4]] = 1
        features['col4', state[3:17:4]] = 1
        features['2x21', state[0:2]+state[4:6]] = 1
        features['2x22', state[1:3]+state[5:7]] = 1
        features['2x23', state[2:4]+state[6:8]] = 1
        features['2x24', state[4:6]+state[8:10]] = 1
        features['2x25', state[5:7]+state[9:11]] = 1
        features['2x26', state[6:8]+state[10:12]] = 1
        features['2x27', state[8:10]+state[12:14]] = 1
        features['2x28', state[9:11]+state[13:15]] = 1
        features['2x29', state[10:12]+state[14:16]] = 1

        return features

