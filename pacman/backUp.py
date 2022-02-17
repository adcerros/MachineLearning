
class BasicAgentAA(BustersAgent):
    followingBFS = False
    currentBFS = []
    POSSIBLE_ACTIONS_NUM = 4
    currentNearlyGhostIndex = None
    nearlyGhostPos = None

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal newStates for Pacman in currentState position
        print("Legal newStates: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())
        
        
    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        livingGhosts = gameState.getLivingGhosts()
        if self.currentNearlyGhostIndex != None: 
            if len(self.currentBFS) < 1 or livingGhosts[self.currentNearlyGhostIndex + 1] == False:
                self.currentNearlyGhostIndex = None
                self.followingBFS = False
        if self.currentNearlyGhostIndex == None:
            self.getNearlyGhostPos(gameState)
        if self.followingBFS == False:
            return self.getNextMove(gameState)
        if self.followingBFS == True:
            return self.currentBFS.pop(0)



    def getNearlyGhostPos(self, gameState):
        # nealyGhost = [manhattanDistance, ghostNumber]
        nearlyGhost = [1000, 0]
        livingGhosts = gameState.getLivingGhosts()
        for i in range(len(gameState.data.ghostDistances)):
            if livingGhosts[i + 1] and gameState.data.ghostDistances[i] != None :
                if gameState.data.ghostDistances[i] < nearlyGhost[0]:
                    nearlyGhost = [gameState.data.ghostDistances[i], i]
        self.currentNearlyGhostIndex = nearlyGhost[1]
       
    def getNextMove(self, gameState):
        nextMoves = []
        pacmanPosition = gameState.getPacmanPosition()
        ghostsPositions = gameState.getGhostPositions()
        self.nearlyGhostPos = ghostsPositions[self.currentNearlyGhostIndex]
        walls = gameState.getWalls()
        # nextMove = [nextPosition, distanceToTarget, action]
        nextMoves.append([(pacmanPosition[0] + 1, pacmanPosition[1]), (abs(self.nearlyGhostPos[0] - (pacmanPosition[0] + 1)) + abs(self.nearlyGhostPos[1] - pacmanPosition[1])), Directions.EAST])
        nextMoves.append([(pacmanPosition[0] - 1, pacmanPosition[1]), (abs(self.nearlyGhostPos[0] - (pacmanPosition[0] - 1)) + abs(self.nearlyGhostPos[1] - pacmanPosition[1])),  Directions.WEST])
        nextMoves.append([(pacmanPosition[0], pacmanPosition[1] + 1), (abs(self.nearlyGhostPos[0] - pacmanPosition[0]) + abs(self.nearlyGhostPos[1] - (pacmanPosition[1] + 1))), Directions.NORTH])
        nextMoves.append([(pacmanPosition[0], pacmanPosition[1] - 1), (abs(self.nearlyGhostPos[0] - pacmanPosition[0]) + abs(self.nearlyGhostPos[1] - (pacmanPosition[1] - 1))), Directions.SOUTH])
        bestMove = None
        bestDistance = 10000
        for i in range(len(nextMoves)):
            if nextMoves[i][1] < bestDistance:
                bestDistance = nextMoves[i][1]
                bestMove = nextMoves[i]
        # if  walls[bestMove[0][0]][bestMove[0][1]] == False:
        #     return bestMove[2]
        # else:
        return self.searchBFS(gameState)


    def searchBFS(self, gameState):
        openList = []
        closedList = []
        pacmanPosition = gameState.getPacmanPosition()
        walls = gameState.getWalls()
        initial = [pacmanPosition, [], 100]
        openList.append(initial)
        while len(openList) > 0:
            currentState = openList.pop(0)
            if currentState[0] == self.nearlyGhostPos:
                self.currentBFS = currentState[1]
                self.followingBFS = True
                return self.currentBFS.pop(0)
            else:
                self.expandStates(walls, openList, closedList, currentState)
                closedList.append(currentState[0])
        print("No solution problem !!!!")
        return Directions.STOP

    def expandStates(self, walls, openList, closedList, currentState):
        newStates = []
        newStates.append([(currentState[0][0], currentState[0][1] + 1), copy.deepcopy(currentState[1]) + [Directions.NORTH]])
        newStates.append([(currentState[0][0], currentState[0][1] - 1), copy.deepcopy(currentState[1]) + [Directions.SOUTH]])
        newStates.append([(currentState[0][0] - 1, currentState[0][1]), copy.deepcopy(currentState[1]) + [Directions.WEST]])
        newStates.append([(currentState[0][0] + 1, currentState[0][1]), copy.deepcopy(currentState[1]) + [Directions.EAST]])
        for i in range(self.POSSIBLE_ACTIONS_NUM):
            currentNewState = newStates.pop(0)
            if walls[currentNewState[0][0]][currentNewState[0][1]] == False and currentNewState not in openList and currentNewState[0] not in closedList:
                self.insertNewState(openList, currentNewState)

    #Version con heuristica
    def insertNewState(self, openList, currentNewState):
        funcG = (abs(self.nearlyGhostPos[0] - currentNewState[0][0]) + abs(self.nearlyGhostPos[1] - currentNewState[0][1]) + len(currentNewState[1]))
        currentNewState.append(funcG)
        for j in range(len(openList)):
            if (funcG <= openList[j][2]):
                openList.insert(j, currentNewState)
                return
        openList.append(currentNewState)


    def printLineData(self, gameState):
        return (str(gameState.getPacmanPosition()) + ", " +  str(self.countFood(gameState))  + ", " +  str(gameState.getGhostPositions()) + ", " +  str(gameState.getLivingGhosts()))
