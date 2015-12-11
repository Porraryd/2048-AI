import collections
import game
import copy, random
import pickle

import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

LEARNING_RATE = 0.01
c = 15


class TDAgent() :
    def __init__(self) :
        self.row1weight = np.zeros(pow(c,4))
        self.row2weight = np.zeros(pow(c,4))
        self.row3weight = np.zeros(pow(c,4))
        self.row4weight = np.zeros(pow(c,4))
        self.col1weight = np.zeros(pow(c,4))
        self.col2weight = np.zeros(pow(c,4))
        self.col3weight = np.zeros(pow(c,4))
        self.col4weight = np.zeros(pow(c,4))

        self.square1weight = np.zeros(pow(c,4))
        self.square2weight = np.zeros(pow(c,4))
        self.square3weight = np.zeros(pow(c,4))
        self.square4weight = np.zeros(pow(c,4))
        self.square5weight = np.zeros(pow(c,4))
        self.square6weight = np.zeros(pow(c,4))
        self.square7weight = np.zeros(pow(c,4))
        self.square8weight = np.zeros(pow(c,4))
        self.square9weight = np.zeros(pow(c,4))


    #Chooses the most beneficial move and returns it
    #If tie, choose at random
    def chooseMove(self, state, possibleMoves) : 

        #If there is only one move, just return it
        if len(possibleMoves) == 1 :
            return tuple(possibleMoves)[0]

        #Calculate possible moves and their middlestate
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


    def updateFeatures(self, state, expectedVal) :

        val = self.V(state)

        error = (expectedVal - val) * LEARNING_RATE

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


class TDMiddlestateAgent() :
    def __init__(self) :
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
        #Calculate possible moves and their middlestate
        moves = []
        for i in possibleMoves :
            moves.append((self.evaluate(state, i),i))

        ##Return randomly between the best moves
        maxR = max(moves)[0]
        bestMoves = [a for (r,a) in moves if isclose(r, maxR)]

        return random.choice(bestMoves) 
    
    def computeMiddlestate(self, state, action) :
            return move(state,action)

    def evaluate(self, state, action) :
            (reward, afterstate) = self.computeMiddlestate(state, action)
            return reward + self.V(afterstate)

    #Takes a state, action, reward and learns from it using TD-learning
    def learnEval(self, state, action, reward, newState, afterState) :
        moves = []
        for i in range(4) :
            (reward, nexts) = self.computeMiddlestate(newState, i)
            if (nexts != newState) :
                moves.append((reward + self.V(nexts), i))
        if not moves :
            return
        aNext = max(moves)

        self.updateFeatures(afterState, aNext[0])
        


    def saveWeights(self):
        # print 'self. s',self.weights
        np.savez("agents/TD", self.row1weight,self.row2weight,self.row3weight,self.row4weight,\
                        self.col1weight,self.col2weight,self.col3weight,self.col4weight,self.square1weight,\
                        self.square2weight,self.square3weight,self.square4weight,self.square5weight,self.square6weight,\
                        self.square7weight,self.square8weight,self.square9weight)



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

    #Calculate the estimated value of a state using the current weight vector
    def V(self, state) :
        return self.sumFeatures(state)

    def sumFeatures(self, state) :
        sumV = 0

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

    def updateFeatures(self, state, expectedVal) :

        val = self.V(state)

        error = (expectedVal - val) * LEARNING_RATE

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
