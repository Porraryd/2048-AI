import collections
import game
import copy, random
import pickle

import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

LEARNING_RATE = 0.025


class TDAgent() :
    def __init__(self) :
        self.weights = collections.Counter()
        self.featureExtractor = featureExtractor

        self.row1weight = np.zeros(pow(15,4))
        self.row2weight = np.zeros(pow(15,4))
        self.row3weight = np.zeros(pow(15,4))
        self.row4weight = np.zeros(pow(15,4))
        self.col1weight = np.zeros(pow(15,4))
        self.col2weight = np.zeros(pow(15,4))
        self.col3weight = np.zeros(pow(15,4))
        self.col4weight = np.zeros(pow(15,4))

        self.square1weight = np.zeros(pow(15,4))
        self.square2weight = np.zeros(pow(15,4))
        self.square3weight = np.zeros(pow(15,4))
        self.square4weight = np.zeros(pow(15,4))
        self.square5weight = np.zeros(pow(15,4))
        self.square6weight = np.zeros(pow(15,4))
        self.square7weight = np.zeros(pow(15,4))
        self.square8weight = np.zeros(pow(15,4))
        self.square9weight = np.zeros(pow(15,4))


    #Chooses the most beneficial move and returns it
    #If tie, choose at random
    def chooseMove(self, state, possibleMoves) : 

        #If there is only one move, just return it
        if len(possibleMoves) == 1 :
            return tuple(possibleMoves)[0]

        #Calculate possible moves and their afterstate
        moves = []

        for i in possibleMoves :
            (reward, newState) = move(state, i)
            if newState == state :
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
            moves.append((reward + sumV, i))

        ##Return randomly between the best moves
        maxR = max(moves)[0]
        bestMoves = [a for (r,a) in moves if isclose(r, maxR)]

        return random.choice(bestMoves)
        return max(moves)[1]


    #Takes a state, action, reward and learns from it using TD-learning
    def learnEval(self, state, action, reward, newState, afterState) :
        newV = self.V(newState)
        V = self.V(state)

        self.updateFeatures(state, reward + newV)


    def saveWeights(self):
        # print 'self. s',self.weights
        np.savez("agents/TDAS", self.row1weight,self.row2weight,self.row3weight,self.row4weight,\
                        self.col1weight,self.col2weight,self.col3weight,self.col4weight,self.square1weight,\
                        self.square2weight,self.square3weight,self.square4weight,self.square5weight,self.square6weight,\
                        self.square7weight,self.square8weight,self.square9weight)

    def loadWeights(self):
        arrays = np.load("agents/TDAS.npz")
        
        self.row1weight = arrays['arr_0']
        self.row2weight = arrays['arr_1']
        self.row3weight = arrays['arr_2']
        self.row4weight = arrays['arr_3']
        self.col1weight = arrays['arr_4']
        self.col2weight = arrays['arr_5']
        self.col3weight = arrays['arr_6']
        self.col4weight = arrays['arr_7']

        self.square1weight = arrays['arr_8']
        self.square2weight = arrays['arr_9']
        self.square3weight = arrays['arr_10']
        self.square4weight = arrays['arr_11']
        self.square5weight = arrays['arr_12']
        self.square6weight = arrays['arr_13']
        self.square7weight = arrays['arr_14']
        self.square8weight = arrays['arr_15']
        self.square9weight = arrays['arr_16']   


    #Calculate the estimated value of a state using the current weight vector
    def V(self, state) :
        return self.sumFeatures(state)


    def updateFeatures(self, state, float expectedVal) :

        val = self.V(state)

        error = (expectedVal - val) * LEARNING_RATE

        c = 15
        self.row1weight[state[0] + c*state[1] + c*c*state[2]+ c*c*c*state[3]] += error
        self.row2weight[state[4] + c*state[5] + c*c*state[6]+ c*c*c*state[7]] += error
        self.row3weight[state[8] + c*state[9] + c*c*state[10]+ c*c*c*state[11]] += error
        self.row4weight[state[12] + c*state[13] + c*c*state[14]+ c*c*c*state[15]] += error

        self.col1weight[state[0] + c*state[4] + c*c*state[8]+ c*c*c*state[12]] += error
        self.col2weight[state[1] + c*state[5] + c*c*state[9]+ c*c*c*state[13]] += error
        self.col3weight[state[2] + c*state[6] + c*c*state[10]+ c*c*c*state[14]] += error
        self.col4weight[state[3] + c*state[7] + c*c*state[11]+ c*c*c*state[15]] += error

        self.square1weight[state[0] + c*state[1] + c*c*state[4]+ c*c*c*state[5]] += error
        self.square2weight[state[1] + c*state[2] + c*c*state[5]+ c*c*c*state[6]] += error
        self.square3weight[state[2] + c*state[3] + c*c*state[6]+ c*c*c*state[7]] += error
        self.square4weight[state[4] + c*state[5] + c*c*state[8]+ c*c*c*state[9]] += error
        self.square5weight[state[5] + c*state[6] + c*c*state[9]+ c*c*c*state[10]] += error
        self.square6weight[state[6] + c*state[7] + c*c*state[10]+ c*c*c*state[11]] += error
        self.square7weight[state[8] + c*state[9] + c*c*state[12]+ c*c*c*state[13]] += error
        self.square8weight[state[9] + c*state[10] + c*c*state[13]+ c*c*c*state[14]] += error
        self.square9weight[state[10] + c*state[11] + c*c*state[14]+ c*c*c*state[15]] += error

        
    def sumFeatures(self, state) :
        sumV = 0

        c = 15
        sumV += self.row1weight[state[0] + c*state[1] + c*c*state[2]+ c*c*c*state[3]]
        sumV += self.row2weight[state[4] + c*state[5] + c*c*state[6]+ c*c*c*state[7]]
        sumV += self.row3weight[state[8] + c*state[9] + c*c*state[10]+ c*c*c*state[11]]
        sumV += self.row4weight[state[12] + c*state[13] + c*c*state[14]+ c*c*c*state[15]]

        sumV += self.col1weight[state[0] + c*state[4] + c*c*state[8]+ c*c*c*state[12]]
        sumV += self.col2weight[state[1] + c*state[5] + c*c*state[9]+ c*c*c*state[13]]
        sumV += self.col3weight[state[2] + c*state[6] + c*c*state[10]+ c*c*c*state[14]]
        sumV += self.col4weight[state[3] + c*state[7] + c*c*state[11]+ c*c*c*state[15]]

        sumV += self.square1weight[state[0] + c*state[1] + c*c*state[4]+ c*c*c*state[5]]
        sumV += self.square2weight[state[1] + c*state[2] + c*c*state[5]+ c*c*c*state[6]]
        sumV += self.square3weight[state[2] + c*state[3] + c*c*state[6]+ c*c*c*state[7]]
        sumV += self.square4weight[state[4] + c*state[5] + c*c*state[8]+ c*c*c*state[9]]
        sumV += self.square5weight[state[5] + c*state[6] + c*c*state[9]+ c*c*c*state[10]]
        sumV += self.square6weight[state[6] + c*state[7] + c*c*state[10]+ c*c*c*state[11]]
        sumV += self.square7weight[state[8] + c*state[9] + c*c*state[12]+ c*c*c*state[13]]
        sumV += self.square8weight[state[9] + c*state[10] + c*c*state[13]+ c*c*c*state[14]]
        sumV += self.square9weight[state[10] + c*state[11] + c*c*state[14]+ c*c*c*state[15]]

        return sumV


class TDAfterstateAgent() :
    def __init__(self) :
        self.weights = collections.Counter()
        self.featureExtractor = featureExtractor


        self.row1weight = np.zeros(pow(15,4))
        self.row2weight = np.zeros(pow(15,4))
        self.row3weight = np.zeros(pow(15,4))
        self.row4weight = np.zeros(pow(15,4))
        self.col1weight = np.zeros(pow(15,4))
        self.col2weight = np.zeros(pow(15,4))
        self.col3weight = np.zeros(pow(15,4))
        self.col4weight = np.zeros(pow(15,4))

        self.square1weight = np.zeros(pow(15,4))
        self.square2weight = np.zeros(pow(15,4))
        self.square3weight = np.zeros(pow(15,4))
        self.square4weight = np.zeros(pow(15,4))
        self.square5weight = np.zeros(pow(15,4))
        self.square6weight = np.zeros(pow(15,4))
        self.square7weight = np.zeros(pow(15,4))
        self.square8weight = np.zeros(pow(15,4))
        self.square9weight = np.zeros(pow(15,4))

    #Chooses the most beneficial move and returns it
    #If tie, choose at random
    def chooseMove(self, state, possibleMoves) : 
        #Calculate possible moves and their afterstates
                #If there is only one move, just return it
        if len(possibleMoves) == 1 :
            return tuple(possibleMoves)[0]

        exploration = 0.00
        if (random.random() < exploration) :
            return random.choice(tuple(possibleMoves))
        #Calculate possible moves and their afterstate
        moves = []
        for i in possibleMoves :
            moves.append((self.evaluate(state, i),i))

        ##Return randomly between the best moves
        maxR = max(moves)[0]
        bestMoves = [a for (r,a) in moves if isclose(r, maxR)]

        return random.choice(bestMoves) 
    
    def computeAfterstate(self, state, action) :
            return move(state,action)

    def evaluate(self, state, action) :
            (reward, afterstate) = self.computeAfterstate(state, action)
            return reward + self.V(afterstate)

    #Takes a state, action, reward and learns from it using TD-learning
    def learnEval(self, state, action, reward, newState, afterState) :
        moves = []
        for i in range(4) :
            (reward, afterstate) = self.computeAfterstate(newState, i)
            if (afterstate != newState) :
                moves.append((reward + self.V(afterstate), i, afterstate))
        if not moves :
            return
        aNext = max(moves)
        sNext = aNext[2]
        newV = self.V(sNext)
        rNext = aNext[0] - newV

        self.updateFeatures(afterState, newV + rNext)
        
        # print 'self.weights ', self.weights
        #for f, v in self.featureExtractor(afterState).iteritems():
        #    self.weights[f] = self.weights[f] - (LEARNING_RATE*(V-(rNext+(1*newV)))*v)


    def saveWeights(self):
        # print 'self. s',self.weights
        np.savez("agents/TD", self.row1weight,self.row2weight,self.row3weight,self.row4weight,\
                        self.col1weight,self.col2weight,self.col3weight,self.col4weight,self.square1weight,\
                        self.square2weight,self.square3weight,self.square4weight,self.square5weight,self.square6weight,\
                        self.square7weight,self.square8weight,self.square9weight)

        #with open('saved_weightsAS.txt', 'w') as file_:
        #    print "Saving weights. Do not close me please."
        #    pickle.dump(self.weights, file_)
        #    print "Saving done."

    def loadWeights(self):
        arrays = np.load("agents/TD.npz")
        
        self.row1weight = arrays['arr_0']
        self.row2weight = arrays['arr_1']
        self.row3weight = arrays['arr_2']
        self.row4weight = arrays['arr_3']
        self.col1weight = arrays['arr_4']
        self.col2weight = arrays['arr_5']
        self.col3weight = arrays['arr_6']
        self.col4weight = arrays['arr_7']

        self.square1weight = arrays['arr_8']
        self.square2weight = arrays['arr_9']
        self.square3weight = arrays['arr_10']
        self.square4weight = arrays['arr_11']
        self.square5weight = arrays['arr_12']
        self.square6weight = arrays['arr_13']
        self.square7weight = arrays['arr_14']
        self.square8weight = arrays['arr_15']
        self.square9weight = arrays['arr_16']
        #with open('saved_weightsAS.txt','r') as myFile:
        #    self.weights = pickle.load(myFile)
        #    print "Loaded: %d weights" % len(self.weights)


    #Calculate the estimated value of a state using the current weight vector
    def V(self, state) :
        return self.sumFeatures(state)

    def sumFeatures(self, state) :
        sumV = 0

        c = 15
        sumV += self.row1weight[state[0] + c*state[1] + c*c*state[2]+ c*c*c*state[3]]
        sumV += self.row2weight[state[4] + c*state[5] + c*c*state[6]+ c*c*c*state[7]]
        sumV += self.row3weight[state[8] + c*state[9] + c*c*state[10]+ c*c*c*state[11]]
        sumV += self.row4weight[state[12] + c*state[13] + c*c*state[14]+ c*c*c*state[15]]

        sumV += self.col1weight[state[0] + c*state[4] + c*c*state[8]+ c*c*c*state[12]]
        sumV += self.col2weight[state[1] + c*state[5] + c*c*state[9]+ c*c*c*state[13]]
        sumV += self.col3weight[state[2] + c*state[6] + c*c*state[10]+ c*c*c*state[14]]
        sumV += self.col4weight[state[3] + c*state[7] + c*c*state[11]+ c*c*c*state[15]]

        sumV += self.square1weight[state[0] + c*state[1] + c*c*state[4]+ c*c*c*state[5]]
        sumV += self.square2weight[state[1] + c*state[2] + c*c*state[5]+ c*c*c*state[6]]
        sumV += self.square3weight[state[2] + c*state[3] + c*c*state[6]+ c*c*c*state[7]]
        sumV += self.square4weight[state[4] + c*state[5] + c*c*state[8]+ c*c*c*state[9]]
        sumV += self.square5weight[state[5] + c*state[6] + c*c*state[9]+ c*c*c*state[10]]
        sumV += self.square6weight[state[6] + c*state[7] + c*c*state[10]+ c*c*c*state[11]]
        sumV += self.square7weight[state[8] + c*state[9] + c*c*state[12]+ c*c*c*state[13]]
        sumV += self.square8weight[state[9] + c*state[10] + c*c*state[13]+ c*c*c*state[14]]
        sumV += self.square9weight[state[10] + c*state[11] + c*c*state[14]+ c*c*c*state[15]]

        return sumV

    def updateFeatures(self, state, float expectedVal) :

        val = self.V(state)

        error = (expectedVal - val) * LEARNING_RATE

        c = 15
        self.row1weight[state[0] + c*state[1] + c*c*state[2]+ c*c*c*state[3]] += error
        self.row2weight[state[4] + c*state[5] + c*c*state[6]+ c*c*c*state[7]] += error
        self.row3weight[state[8] + c*state[9] + c*c*state[10]+ c*c*c*state[11]] += error
        self.row4weight[state[12] + c*state[13] + c*c*state[14]+ c*c*c*state[15]] += error

        self.col1weight[state[0] + c*state[4] + c*c*state[8]+ c*c*c*state[12]] += error
        self.col2weight[state[1] + c*state[5] + c*c*state[9]+ c*c*c*state[13]] += error
        self.col3weight[state[2] + c*state[6] + c*c*state[10]+ c*c*c*state[14]] += error
        self.col4weight[state[3] + c*state[7] + c*c*state[11]+ c*c*c*state[15]] += error

        self.square1weight[state[0] + c*state[1] + c*c*state[4]+ c*c*c*state[5]] += error
        self.square2weight[state[1] + c*state[2] + c*c*state[5]+ c*c*c*state[6]] += error
        self.square3weight[state[2] + c*state[3] + c*c*state[6]+ c*c*c*state[7]] += error
        self.square4weight[state[4] + c*state[5] + c*c*state[8]+ c*c*c*state[9]] += error
        self.square5weight[state[5] + c*state[6] + c*c*state[9]+ c*c*c*state[10]] += error
        self.square6weight[state[6] + c*state[7] + c*c*state[10]+ c*c*c*state[11]] += error
        self.square7weight[state[8] + c*state[9] + c*c*state[12]+ c*c*c*state[13]] += error
        self.square8weight[state[9] + c*state[10] + c*c*state[13]+ c*c*c*state[14]] += error
        self.square9weight[state[10] + c*state[11] + c*c*state[14]+ c*c*c*state[15]] += error











class TDAfterstateSymmetryAgent() :
    def __init__(self) :
        self.weights = collections.Counter()
        self.featureExtractor = featureExtractor


        self.outerRowWeight = np.zeros(pow(15,4))
        self.innerRowWeight = np.zeros(pow(15,4))
        self.corner2x3weight = np.zeros(pow(15,6))
        self.inner2x3weight = np.zeros(pow(15,6))


    #Chooses the most beneficial move and returns it
    #If tie, choose at random
    def chooseMove(self, state, possibleMoves) : 
        #Calculate possible moves and their afterstates
                #If there is only one move, just return it
        if len(possibleMoves) == 1 :
            return tuple(possibleMoves)[0]

        exploration = 0.00
        if (random.random() < exploration) :
            return random.choice(tuple(possibleMoves))
        #Calculate possible moves and their afterstate
        moves = []
        for i in possibleMoves :
            moves.append((self.evaluate(state, i),i))

        ##Return randomly between the best moves
        maxR = max(moves)[0]
        bestMoves = [a for (r,a) in moves if isclose(r, maxR)]

        return random.choice(bestMoves) 
    
    def computeAfterstate(self, state, action) :
            return move(state,action)

    def evaluate(self, state, action) :
            (reward, afterstate) = self.computeAfterstate(state, action)
            return reward + self.V(afterstate)

    #Takes a state, action, reward and learns from it using TD-learning
    def learnEval(self, state, action, reward, newState, afterState) :
        moves = []
        for i in range(4) :
            (reward, afterstate) = self.computeAfterstate(newState, i)
            if (afterstate != newState) :
                moves.append((reward + self.V(afterstate), i, afterstate))
        if not moves :
            return
        aNext = max(moves)
        sNext = aNext[2]
        newV = self.V(sNext)
        rNext = aNext[0] - newV

        self.updateFeatures(afterState, newV + rNext)
        
        # print 'self.weights ', self.weights
        #for f, v in self.featureExtractor(afterState).iteritems():
        #    self.weights[f] = self.weights[f] - (LEARNING_RATE*(V-(rNext+(1*newV)))*v)


    def saveWeights(self):
        # print 'self. s',self.weights
        np.savez("agents/TD", self.outerRowWeight, self.innerRowWeight,self.corner2x3weight, self.inner2x3weight);

        #with open('saved_weightsAS.txt', 'w') as file_:
        #    print "Saving weights. Do not close me please."
        #    pickle.dump(self.weights, file_)
        #    print "Saving done."

    def loadWeights(self):
        arrays = np.load("agents/TD.npz")
        
        self.outerRowWeight = arrays['arr_0']
        self.innerRowWeight = arrays['arr_1']
        self.corner2x3weight = arrays['arr_2']
        self.inner2x3weight = arrays['arr_3']
        #with open('saved_weightsAS.txt','r') as myFile:
        #    self.weights = pickle.load(myFile)
        #    print "Loaded: %d weights" % len(self.weights)


    #Calculate the estimated value of a state using the current weight vector
    def V(self, state) :
        return self.sumFeatures(state)

    def sumFeatures(self, state) :
        sumV = 0
        c=15
        sumV += self.outerRowWeight[state[0]+c*state[1]+c*c*state[2]+(c**3)*state[3]]
        sumV += self.outerRowWeight[state[0]+c*state[4]+c*c*state[8]+(c**3)*state[12]]
        sumV += self.outerRowWeight[state[3]+c*state[7]+c*c*state[11]+(c**3)*state[15]]
        sumV += self.outerRowWeight[state[12]+c*state[13]+c*c*state[14]+(c**3)*state[15]]

        sumV += self.outerRowWeight[state[3]+c*state[2]+c*c*state[1]+(c**3)*state[0]]
        sumV += self.outerRowWeight[state[12]+c*state[8]+c*c*state[4]+(c**3)*state[0]]
        sumV += self.outerRowWeight[state[15]+c*state[11]+c*c*state[7]+(c**3)*state[3]]
        sumV += self.outerRowWeight[state[15]+c*state[14]+c*c*state[13]+(c**3)*state[12]]

        sumV += self.innerRowWeight[state[4]+c*state[5]+c*c*state[6]+(c**3)*state[7]]
        sumV += self.innerRowWeight[state[8]+c*state[9]+c*c*state[10]+(c**3)*state[11]]
        sumV += self.innerRowWeight[state[1]+c*state[5]+c*c*state[9]+(c**3)*state[13]]
        sumV += self.innerRowWeight[state[2]+c*state[6]+c*c*state[10]+(c**3)*state[14]]

        sumV += self.innerRowWeight[state[7]+c*state[6]+c*c*state[5]+(c**3)*state[4]]
        sumV += self.innerRowWeight[state[11]+c*state[10]+c*c*state[9]+(c**3)*state[8]]
        sumV += self.innerRowWeight[state[13]+c*state[9]+c*c*state[5]+(c**3)*state[1]]
        sumV += self.innerRowWeight[state[14]+c*state[10]+c*c*state[6]+(c**3)*state[2]]


        sumV += self.corner2x3weight[state[0]+c*state[1]+c*c*state[4]+(c**3)*state[5]+(c**4)*state[8]+(c**5)*state[9]]
        sumV += self.corner2x3weight[state[3]+c*state[2]+c*c*state[7]+(c**3)*state[6]+(c**4)*state[11]+(c**5)*state[10]]
        sumV += self.corner2x3weight[state[12]+c*state[13]+c*c*state[8]+(c**3)*state[9]+(c**4)*state[4]+(c**5)*state[5]]
        sumV += self.corner2x3weight[state[15]+c*state[14]+c*c*state[11]+(c**3)*state[10]+(c**4)*state[7]+(c**5)*state[6]]

        sumV += self.corner2x3weight[state[0]+c*state[4]+c*c*state[1]+(c**3)*state[5]+(c**4)*state[2]+(c**5)*state[6]]
        sumV += self.corner2x3weight[state[3]+c*state[7]+c*c*state[2]+(c**3)*state[6]+(c**4)*state[1]+(c**5)*state[5]]
        sumV += self.corner2x3weight[state[12]+c*state[8]+c*c*state[13]+(c**3)*state[9]+(c**4)*state[14]+(c**5)*state[10]]
        sumV += self.corner2x3weight[state[15]+c*state[11]+c*c*state[14]+(c**3)*state[10]+(c**4)*state[13]+(c**5)*state[9]]

        sumV += self.inner2x3weight[state[1]+c*state[2]+c*c*state[5]+(c**3)*state[6]+(c**4)*state[9]+(c**5)*state[10]]
        sumV += self.inner2x3weight[state[2]+c*state[1]+c*c*state[6]+(c**3)*state[5]+(c**4)*state[10]+(c**5)*state[9]]

        sumV += self.inner2x3weight[state[13]+c*state[14]+c*c*state[9]+(c**3)*state[10]+(c**4)*state[5]+(c**5)*state[6]]
        sumV += self.inner2x3weight[state[14]+c*state[13]+c*c*state[10]+(c**3)*state[9]+(c**4)*state[6]+(c**5)*state[5]]

        sumV += self.inner2x3weight[state[4]+c*state[8]+c*c*state[5]+(c**3)*state[9]+(c**4)*state[6]+(c**5)*state[10]]
        sumV += self.inner2x3weight[state[8]+c*state[4]+c*c*state[9]+(c**3)*state[5]+(c**4)*state[10]+(c**5)*state[6]]

        sumV += self.inner2x3weight[state[7]+c*state[11]+c*c*state[6]+(c**3)*state[10]+(c**4)*state[5]+(c**5)*state[9]]
        sumV += self.inner2x3weight[state[11]+c*state[7]+c*c*state[10]+(c**3)*state[6]+(c**4)*state[9]+(c**5)*state[5]]
        
        return sumV

    def updateFeatures(self, state, float expectedVal) :

        val = self.V(state)

        error = (expectedVal - val) * LEARNING_RATE

        c=15
        self.outerRowWeight[state[0]+c*state[1]+c*c*state[2]+(c**3)*state[3]] += error
        self.outerRowWeight[state[0]+c*state[4]+c*c*state[8]+(c**3)*state[12]]+= error
        self.outerRowWeight[state[3]+c*state[7]+c*c*state[11]+(c**3)*state[15]]+= error
        self.outerRowWeight[state[12]+c*state[13]+c*c*state[14]+(c**3)*state[15]]+= error

        self.outerRowWeight[state[3]+c*state[2]+c*c*state[1]+(c**3)*state[0]]+= error
        self.outerRowWeight[state[12]+c*state[8]+c*c*state[4]+(c**3)*state[0]]+= error
        self.outerRowWeight[state[15]+c*state[11]+c*c*state[7]+(c**3)*state[3]]+= error
        self.outerRowWeight[state[15]+c*state[14]+c*c*state[13]+(c**3)*state[12]]+= error

        self.innerRowWeight[state[4]+c*state[5]+c*c*state[6]+(c**3)*state[7]]+= error
        self.innerRowWeight[state[8]+c*state[9]+c*c*state[10]+(c**3)*state[11]]+= error
        self.innerRowWeight[state[1]+c*state[5]+c*c*state[9]+(c**3)*state[13]]+= error
        self.innerRowWeight[state[2]+c*state[6]+c*c*state[10]+(c**3)*state[14]]+= error

        self.innerRowWeight[state[7]+c*state[6]+c*c*state[5]+(c**3)*state[4]]+= error
        self.innerRowWeight[state[11]+c*state[10]+c*c*state[9]+(c**3)*state[8]]+= error
        self.innerRowWeight[state[13]+c*state[9]+c*c*state[5]+(c**3)*state[1]]+= error
        self.innerRowWeight[state[14]+c*state[10]+c*c*state[6]+(c**3)*state[2]]+= error


        self.corner2x3weight[state[0]+c*state[1]+c*c*state[4]+(c**3)*state[5]+(c**4)*state[8]+(c**5)*state[9]]+= error
        self.corner2x3weight[state[3]+c*state[2]+c*c*state[7]+(c**3)*state[6]+(c**4)*state[11]+(c**5)*state[10]]+= error
        self.corner2x3weight[state[12]+c*state[13]+c*c*state[8]+(c**3)*state[9]+(c**4)*state[4]+(c**5)*state[5]]+= error
        self.corner2x3weight[state[15]+c*state[14]+c*c*state[11]+(c**3)*state[10]+(c**4)*state[7]+(c**5)*state[6]]+= error

        self.corner2x3weight[state[0]+c*state[4]+c*c*state[1]+(c**3)*state[5]+(c**4)*state[2]+(c**5)*state[6]]+= error
        self.corner2x3weight[state[3]+c*state[7]+c*c*state[2]+(c**3)*state[6]+(c**4)*state[1]+(c**5)*state[5]]+= error
        self.corner2x3weight[state[12]+c*state[8]+c*c*state[13]+(c**3)*state[9]+(c**4)*state[14]+(c**5)*state[10]]+= error
        self.corner2x3weight[state[15]+c*state[11]+c*c*state[14]+(c**3)*state[10]+(c**4)*state[13]+(c**5)*state[9]]+= error

        self.inner2x3weight[state[1]+c*state[2]+c*c*state[5]+(c**3)*state[6]+(c**4)*state[9]+(c**5)*state[10]]+= error
        self.inner2x3weight[state[2]+c*state[1]+c*c*state[6]+(c**3)*state[5]+(c**4)*state[10]+(c**5)*state[9]]+= error

        self.inner2x3weight[state[13]+c*state[14]+c*c*state[9]+(c**3)*state[10]+(c**4)*state[5]+(c**5)*state[6]]+= error
        self.inner2x3weight[state[14]+c*state[13]+c*c*state[10]+(c**3)*state[9]+(c**4)*state[6]+(c**5)*state[5]]+= error

        self.inner2x3weight[state[4]+c*state[8]+c*c*state[5]+(c**3)*state[9]+(c**4)*state[6]+(c**5)*state[10]]+= error
        self.inner2x3weight[state[8]+c*state[4]+c*c*state[9]+(c**3)*state[5]+(c**4)*state[10]+(c**5)*state[6]]+= error

        self.inner2x3weight[state[7]+c*state[11]+c*c*state[6]+(c**3)*state[10]+(c**4)*state[5]+(c**5)*state[9]]+= error
        self.inner2x3weight[state[11]+c*state[7]+c*c*state[10]+(c**3)*state[6]+(c**4)*state[9]+(c**5)*state[5]]+= error




























#Features extractor splitting the game board into smaller tiles
def featureExtractor(state) :
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

    #highest tile
    features['highest tile in', state.index(max(state))] = 1
    return features

def symmetryFeatureExtractor(state) :
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
    '''

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
    return features;

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


def moveUp(state) :
    BOARD_SIZE = 4
    score = 0
    newState = copy.copy(state)
    for x in range(BOARD_SIZE) :
        merged = False
        emptyY = 0
        for y in range(BOARD_SIZE) :
            val = newState[x + BOARD_SIZE*y]
            if val == 0 :
                continue
            if emptyY > 0 and not merged and (newState[x + BOARD_SIZE*(emptyY-1)] == val) :
                newState[x + BOARD_SIZE * (emptyY-1)] = val+1
                newState[x + BOARD_SIZE * y] = 0
                score += pow(2,val+1)
                merged = True
            else :
                newState[x + BOARD_SIZE * y] = 0
                newState[x + BOARD_SIZE * emptyY]= val
                emptyY += 1
                merged = False

    return (score, newState)

def move(state, move) :
    state = list(state)
    score = 0
    if move == RIGHT :
        rotate(state)
        rotate(state)
        rotate(state)
        (score, state) = moveUp(state)
        rotate(state)
    elif move == DOWN :
        rotate(state)
        rotate(state)
        (score, state) = moveUp(state)
        rotate(state)
        rotate(state)
    elif move == LEFT :
        rotate(state)
        (score, state) = moveUp(state)
        rotate(state)
        rotate(state)
        rotate(state)

    elif move == UP :
        (score, state) = moveUp(state)

    return (score, tuple(state))

def rotate(state) :
        tempState = copy.copy(state)
        state[0] = tempState[12]
        state[1] = tempState[8]
        state[2] = tempState[4]
        state[3] = tempState[0]

        state[4] = tempState[13]
        state[5] = tempState[9]
        state[6] = tempState[5]
        state[7] = tempState[1]

        state[8] = tempState[14]
        state[9] = tempState[10]
        state[10] = tempState[6]
        state[11] = tempState[2]

        state[12] = tempState[15]
        state[13] = tempState[11]
        state[14] = tempState[7]
        state[15] = tempState[3]
