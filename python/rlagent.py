import collections
import game
import copy, random
import pickle

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class RLAgent :
    #Chooses the most beneficial move and returns it
    #If tie, choose at random
    def chooseMove(self, state) : raise NotImplementedError("Override me")

    #Takes a state, action, reward and learns from it using TD-learning
    def learnEval(self, state, action, reward, newState) : raise NotImplementedError("Override me")


class TDAgent() :
    def __init__(self) :
        self.weights = collections.Counter()

    #Chooses the most beneficial move and returns it
    #If tie, choose at random
    def chooseMove(self, state) : 
        #Calculate possible moves and their afterstate
        moves = []

        for i in range(4) :
            newGame = game.Game(state, True)
            newGame.move(i)
            newState = newGame.getState()
            if newState != state : 
                moves.append((newGame.score + self.V(newState), i))

        return max(moves)[1]


    #Takes a state, action, reward and learns from it using TD-learning
    def learnEval(self, state, action, reward, newState) :
        newV = self.V(newState)
        V = self.V(state)
        # print 'self.weights ', self.weights
        for f, v in self.featureExtractor(state).iteritems():
            self.weights[f] = self.weights[f] - 0.007*(V-(reward+(1*newV)))*v 


    def saveWeights(self):
        # print 'self. s',self.weights
        with open('saved_weights.txt', 'w') as file_:
            pickle.dump(self.weights, file_)

    def loadWeights(self):
        with open('saved_weights.txt','r') as myFile:
            self.weights = pickle.load(myFile)


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
