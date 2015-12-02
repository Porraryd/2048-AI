import sys,random
import collections

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class Game:

    def __init__(self, startState=None) :
        self.BOARD_SIZE = 4
        self.START_TILES = 2
        self.grid = Grid(self.BOARD_SIZE, startState)
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
        self.grid = Grid(self.BOARD_SIZE)
        self.score = 0
        self.addStartTiles()
        self.grid.setPreState(self.getState())

    def addRandomTile(self) :
        if self.grid.cellEmpty() :
            (x,y) = self.grid.getEmptyCell()
            value = 1 if random.random() < 0.9 else 2
            self.grid.setCell(x,y,value)

    def addStartTiles(self) : 
        for i in range(self.START_TILES) :
            self.addRandomTile()

    def move(self, move) :
        moved = False;

        #Determine travelsal order: 
        xTrav = range(self.BOARD_SIZE) 
        yTrav = range(self.BOARD_SIZE)
        xd =0
        yd =0
        if move == RIGHT :
            xTrav = xTrav[::-1]
            xd = 1
        elif move == DOWN :
            yTrav = yTrav[::-1]
            yd = 1
        elif move == LEFT :
            xd = -1
        elif move == UP :
            yd= -1

        mergedTile = collections.Counter()
        for x in xTrav : 
            for y in yTrav : 
                val = self.grid.getCell(x,y)
                if (val != 0) : 
                    #Start by moving the cell as far as possible
                    curCell = (x,y)
                    nextCell = (x+xd, y+yd)
                    while (self.cellInBounds(nextCell[0],nextCell[1]) and
                        self.grid.getCell(nextCell[0],nextCell[1]) == 0) :
                        curCell = nextCell
                        nextCell = (nextCell[0]+xd, nextCell[1]+yd)

                    self.grid.setCell(curCell[0],curCell[1], val)
                    if curCell != (x,y) :
                        moved = True
                        self.grid.setCell(x,y,0) 

                    #Check for merge: 
                    if (self.cellInBounds(nextCell[0],nextCell[1]) and
                        mergedTile[nextCell] == 0 and
                        self.grid.getCell(nextCell[0],nextCell[1]) == val) :
                        self.grid.setCell(nextCell[0],nextCell[1], val +1)
                        self.grid.setCell(curCell[0],curCell[1], 0)
                        moved = True
                        mergedTile[nextCell] = 1
                        self.score += pow(2,val+1)

        #If anything happened to the board, add a random tile 
        if moved :
            self.grid.setPreState(self.grid.getState())
            self.addRandomTile()

    def cellInBounds(self, x, y) :
        return (x > -1 and x < self.BOARD_SIZE and \
                y > -1 and y < self.BOARD_SIZE )

    def mergeExists(self) :
        for x in range(self.BOARD_SIZE) :
            for y in range(self.BOARD_SIZE) :
                val = self.grid.getCell(x,y)

                if (x != self.BOARD_SIZE-1  and self.grid.getCell(x+1, y) == val) :
                    return True
                if (x != 0 and self.grid.getCell(x-1, y) == val) :
                    return True
                if (y != 0 and self.grid.getCell(x, y-1) == val) :
                    return True
                if (y != self.BOARD_SIZE-1  and self.grid.getCell(x, y+1) == val) :
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

