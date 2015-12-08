import sys,random
import collections, copy

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

BOARD_SIZE = 4
START_TILES = 2

class Game:

    def __init__(self, startState=None) :
        self.grid = Grid(BOARD_SIZE, startState)
        if startState == None:
            self.addStartTiles()
        self.score = 0
        
    def getState(self) :
        return self.grid.getState()

    #Returns the state BEFORE the random tile was added. 
    def getPreState(self) :
        return self.grid.getPreState();

    def printBoard(self) :
        self.grid.printGrid()

    def reset(self) :
        self.grid = Grid(BOARD_SIZE)
        self.score = 0
        self.addStartTiles()

    def addRandomTile(self) :
        if self.grid.cellEmpty() :
            (x,y) = self.grid.getEmptyCell()
            value = 1 if random.random() < 0.9 else 2
            self.grid.setCell(x,y,value)

    def addStartTiles(self) : 
        for i in range(START_TILES) :
            self.addRandomTile()

    def move(self, move) :

        if move == RIGHT :
            self.grid.rotate()
            self.grid.rotate()
            self.grid.rotate()
            self.moveUp()
            self.grid.rotate()
        elif move == DOWN :
            self.grid.rotate()
            self.grid.rotate()
            self.moveUp()
            self.grid.rotate()
            self.grid.rotate()
        elif move == LEFT :
            self.grid.rotate()
            self.moveUp()
            self.grid.rotate()
            self.grid.rotate()
            self.grid.rotate()

        elif move == UP :
            self.moveUp()

        moved = True
        if moved :
            self.grid.setPreState(self.grid.getState())
            self.addRandomTile()

    def moveUp(self) :

        for x in range(BOARD_SIZE) :
            merged = False
            emptyY = 0
            for y in range(BOARD_SIZE) :
                val = self.grid.getCell(x,y)
                if val == 0 :
                    continue
                if emptyY > 0 and not merged and (self.grid.getCell(x,emptyY-1) == val) :
                    self.grid.setCell(x, emptyY-1, val+1)
                    self.grid.setCell(x,y,0)
                    self.score += pow(2,val+1)
                    merged = True
                else :
                    self.grid.setCell(x,y,0)
                    self.grid.setCell(x,emptyY,val)
                    emptyY += 1
                    merged = False


    def cellInBounds(self, x, y) :
        return (x > -1 and x < BOARD_SIZE and \
                y > -1 and y < BOARD_SIZE )


    def getPossibleMoves(self) :
        moves = []

        for x in range(BOARD_SIZE) :
            for y in range(BOARD_SIZE) :
                if self.grid.getCell(x,y) != 0 :
                    continue
                if UP not in moves : 
                    for y2 in range (y+1, BOARD_SIZE) :
                        if (self.grid.getCell(x,y2) != 0) :
                            moves.append(UP)
                            break
                if DOWN not in moves : 
                    for y2 in range (y) :
                        if (self.grid.getCell(x,y2) != 0) :
                            moves.append(DOWN)
                            break
                if LEFT not in moves : 
                    for x2 in range (x+1, BOARD_SIZE) :
                        if (self.grid.getCell(x2,y) != 0) :
                            moves.append(LEFT)
                            break
                if RIGHT not in moves : 
                    for x2 in range (x) :
                        if (self.grid.getCell(x2,y) != 0) :
                            moves.append(RIGHT)
                            break
        if len(moves) == 4:
            return set(moves)
        if LEFT not in moves or RIGHT not in moves :  
            for x in range(BOARD_SIZE-1) :
                for y in range(BOARD_SIZE) :

                    if (self.grid.getCell(x,y) == self.grid.getCell(x+1,y)) and self.grid.getCell(x,y) != 0 :
                        moves.append(LEFT)
                        moves.append(RIGHT)
                        break

        if UP not in moves or DOWN not in moves :  
            for x in range(BOARD_SIZE) :
                for y in range(BOARD_SIZE-1) :

                    if (self.grid.getCell(x,y) == self.grid.getCell(x,y+1)) and self.grid.getCell(x,y) != 0 :
                        moves.append(UP)
                        moves.append(DOWN)
                        break
        return set(moves)


    def mergeExists(self) :
        for x in range(BOARD_SIZE) :
            for y in range(BOARD_SIZE) :
                val = self.grid.getCell(x,y)

                if (x != BOARD_SIZE-1  and self.grid.getCell(x+1, y) == val) :
                    return True
                if (x != 0 and self.grid.getCell(x-1, y) == val) :
                    return True
                if (y != 0 and self.grid.getCell(x, y-1) == val) :
                    return True
                if (y != BOARD_SIZE-1  and self.grid.getCell(x, y+1) == val) :
                    return True

        return False

    def gameOver(self) :
        return (not (self.grid.cellEmpty() or self.mergeExists()))

class Grid:
    def __init__(self, size, startState=None) : 
        #Internal representation, 1d array going from top-left. 
        self.GRID_SIZE = size
        if startState == None :
            self.state = [0 for x in range(size*size)]
        else : 
            self.state = list(startState)

        self.oldState = self.state;
    #Returns true if any cell is empty
    def cellEmpty(self) :
        return 0 in self.state

    #Returns the raw 16-size array state representation
    def getState(self) :
        return tuple(self.state)

    #Returns the state BEFORE the random tile was added. 
    def getPreState(self) :
        return tuple(self.oldState)

    def setPreState(self, state) :
        self.oldState = state

    #ROTATION CLOCKWISE
    def rotate(self) :
        tempState = copy.copy(self.state)
        self.state[0] = tempState[12]
        self.state[1] = tempState[8]
        self.state[2] = tempState[4]
        self.state[3] = tempState[0]

        self.state[4] = tempState[13]
        self.state[5] = tempState[9]
        self.state[6] = tempState[5]
        self.state[7] = tempState[1]

        self.state[8] = tempState[14]
        self.state[9] = tempState[10]
        self.state[10] = tempState[6]
        self.state[11] = tempState[2]

        self.state[12] = tempState[15]
        self.state[13] = tempState[11]
        self.state[14] = tempState[7]
        self.state[15] = tempState[3]
    #Get a random empty cell, returns (x,y)
    def getEmptyCell(self) :
        indices = [i for i in range(self.GRID_SIZE*self.GRID_SIZE) if self.state[i] == 0]
        index = random.choice(indices)
        return (index % self.GRID_SIZE, index / self.GRID_SIZE)

    def getCell(self, x, y) : 
        return self.state[x+y*self.GRID_SIZE]

    def setCell(self,x,y,val) :
        self.state[x+y*self.GRID_SIZE] = val;

    def printGrid(self) :
        for i in range(self.GRID_SIZE) :
            print [cell for cell in self.state[i*self.GRID_SIZE:(i+1)*self.GRID_SIZE]]    

##################################################################
##################################################################
##################################################################

