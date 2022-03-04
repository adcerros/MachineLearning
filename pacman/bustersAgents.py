from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
from logging import root
from re import L
from unittest import result
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
import copy

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True


    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."
    currentAction = "Stop"

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        self.currentAction = KeyboardAgent.getAction(self, gameState)
        return KeyboardAgent.getAction(self, gameState)
    
    # Retorna la posicion del pacman, los fantasmas y su estado (vivo/muerto)
    def printLineData(self, gameState):
        pacmanPosition = gameState.getPacmanPosition()
        ghostPositions = gameState.getGhostPositions()
        livingGhost = gameState.getLivingGhosts()
        legalActions = gameState.getLegalPacmanActions()
        ghostDirections = gameState.getGhostDirections()
        ghostDistances = gameState.data.ghostDistances
        ghost1Distance = 1000
        ghost2Distance = 1000
        ghost3Distance = 1000
        ghost4Distance = 1000
        legalNorth = 0
        legalSouth = 0
        legalWest = 0
        legalEast = 0
        if 'North' in legalActions:
            legalNorth = 1
        if 'South' in legalActions:
            legalSouth = 1
        if 'West' in legalActions:
            legalWest = 1
        if 'East' in legalActions:
            legalEast = 1
        if ghostDistances[0] != None:
            ghost1Distance = ghostDistances[0]
        if ghostDistances[1] != None:
            ghost2Distance = ghostDistances[1]
        if ghostDistances[2] != None:
            ghost3Distance = ghostDistances[2]
        if ghostDistances[3] != None:
            ghost4Distance = ghostDistances[3]
        return  str(pacmanPosition[0]) + ", " + str(pacmanPosition[1]) + ", " + \
                str(ghostPositions[0][0]) + ", "  +  str(ghostPositions[0][1]) + ", " + \
                str(ghostPositions[1][0]) + ", "  +  str(ghostPositions[1][1]) + ", " + \
                str(ghostPositions[2][0]) + ", "  +  str(ghostPositions[2][1]) + ", " + \
                str(ghostPositions[3][0]) + ", "  +  str(ghostPositions[3][1]) + ", " + \
                str(legalNorth) + ", " + str(legalSouth) + ", " + \
                str(legalWest) + ", " + str(legalEast) + ", " + \
                str(ghostDirections.get(0))  + ", " + str(ghostDirections.get(1))  + ", " + \
                str(ghostDirections.get(2))  + ", " + str(ghostDirections.get(3))  + ", " + \
                str(ghost1Distance) + ", " + str(ghost2Distance) + ", " + \
                str(ghost3Distance) + ", " + str(ghost4Distance) + ", " + \
                str(livingGhost[1]) + ", "  +  str(livingGhost[2]) + ", " + \
                str(livingGhost[3]) + ", "  +  str(livingGhost[4]) + ", " + str(self.currentAction)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
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
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
    
    
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):
    followingAstar = False
    currentAstar = []
    POSSIBLE_ACTIONS_NUM = 4
    currentNearlyGhostIndex = None
    nearlyGhostPos = None
    currentMove = "Stop"

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
         
    # Establece el objetivo mas cercano y crea un camino optimo mediante A*
    # El camino calculado se sigue hasta alcanzar la posición indicada
    # Una vez alcanzado el estado final evalua si debe establecer un nuevo objetivo 
    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        livingGhosts = gameState.getLivingGhosts()
        # Se comprueba si el pacman se ha comido al fantasma objetivo o se ha llegado al final del camino establecido por A*
        if self.currentNearlyGhostIndex != None: 
            if len(self.currentAstar) < 1 or livingGhosts[self.currentNearlyGhostIndex + 1] == False:
                self.currentNearlyGhostIndex = None
                self.followingAstar = False
        # Si no se esta siguiendo un objetivo se calcula un camino al mismo
        if self.followingAstar == False:
            return self.getNextMove(gameState)
        # Si se esta siguiendo un camino se continua
        if self.followingAstar == True:
            self.currentMove = self.currentAstar[0]
            return self.currentAstar.pop(0)

    # Establece el objetivo mas cercano a la posicion del pacman en el momento de la llamada
    def getNearlyGhostPos(self, gameState):
        # nealyGhost = [manhattanDistance, ghostNumber]
        nearlyGhost = [1000, 0]
        livingGhosts = gameState.getLivingGhosts()
        for i in range(len(gameState.data.ghostDistances)):
            if livingGhosts[i + 1] and gameState.data.ghostDistances[i] != None :
                if gameState.data.ghostDistances[i] < nearlyGhost[0]:
                    nearlyGhost = [gameState.data.ghostDistances[i], i]
        self.currentNearlyGhostIndex = nearlyGhost[1]
        ghostsPositions = gameState.getGhostPositions()
        self.nearlyGhostPos = ghostsPositions[self.currentNearlyGhostIndex]

    #Establece el objetivo mas cercano y obtiene un camino optimo al mismo mediante A*   
    def getNextMove(self, gameState):
        self.getNearlyGhostPos(gameState)
        return self.searchAstar(gameState)


    def searchAstar(self, gameState):
        # Inicializacion de A*
        openList = []
        closedList = []
        pacmanPosition = gameState.getPacmanPosition()
        walls = gameState.getWalls()
        initial = [pacmanPosition, [], 100]
        openList.append(initial)
        while len(openList) > 0:
            currentState = openList.pop(0)
            if currentState[0] == self.nearlyGhostPos:
                self.currentAstar = currentState[1]
                self.followingAstar = True
                self.currentMove = self.currentAstar[0]
                return self.currentAstar.pop(0)
            else:
                self.expandStates(walls, openList, closedList, currentState)
                closedList.append(currentState[0])
        print("No solution problem !!!!")
        return Directions.STOP

    # Se generan las posibles acciones y los nuevos estados
    def expandStates(self, walls, openList, closedList, currentState):
        newStates = []
        newStates.append([(currentState[0][0], currentState[0][1] + 1), copy.deepcopy(currentState[1]) + [Directions.NORTH]])
        newStates.append([(currentState[0][0], currentState[0][1] - 1), copy.deepcopy(currentState[1]) + [Directions.SOUTH]])
        newStates.append([(currentState[0][0] - 1, currentState[0][1]), copy.deepcopy(currentState[1]) + [Directions.WEST]])
        newStates.append([(currentState[0][0] + 1, currentState[0][1]), copy.deepcopy(currentState[1]) + [Directions.EAST]])
        for i in range(self.POSSIBLE_ACTIONS_NUM):
            currentNewState = newStates.pop(0)
            # Si el nuevo estado no se encuentra repetido
            if walls[currentNewState[0][0]][currentNewState[0][1]] == False and currentNewState not in openList and currentNewState[0] not in closedList:
                self.insertNewState(openList, currentNewState)

    # Se inserta el nuevo estado en la lista abierta en orden segun el valor de su funcion G
    def insertNewState(self, openList, currentNewState):
        # funcG = g(x) + h(x) donde el coste es el numero de pasos del camino y la funcion heuristica se basa en la distancia de Manhattan
        funcG = (abs(self.nearlyGhostPos[0] - currentNewState[0][0]) + abs(self.nearlyGhostPos[1] - currentNewState[0][1]) + len(currentNewState[1]))
        currentNewState.append(funcG)
        for j in range(len(openList)):
            if (funcG <= openList[j][2]):
                openList.insert(j, currentNewState)
                return
        openList.append(currentNewState)

    # Retorna la posicion del pacman, los fantasmas y su estado (vivo/muerto)
    def printLineData(self, gameState):
        pacmanPosition = gameState.getPacmanPosition()
        ghostPositions = gameState.getGhostPositions()
        livingGhost = gameState.getLivingGhosts()
        legalActions = gameState.getLegalPacmanActions()
        ghostDirections = gameState.getGhostDirections()
        ghostDistances = gameState.data.ghostDistances
        ghost1Distance = 1000
        ghost2Distance = 1000
        ghost3Distance = 1000
        ghost4Distance = 1000
        legalNorth = 0
        legalSouth = 0
        legalWest = 0
        legalEast = 0
        if 'North' in legalActions:
            legalNorth = 1
        if 'South' in legalActions:
            legalSouth = 1
        if 'West' in legalActions:
            legalWest = 1
        if 'East' in legalActions:
            legalEast = 1
        if ghostDistances[0] != None:
            ghost1Distance = ghostDistances[0]
        if ghostDistances[1] != None:
            ghost2Distance = ghostDistances[1]
        if ghostDistances[2] != None:
            ghost3Distance = ghostDistances[2]
        if ghostDistances[3] != None:
            ghost4Distance = ghostDistances[3]
        return  str(pacmanPosition[0]) + ", " + str(pacmanPosition[1]) + ", " + \
                str(ghostPositions[0][0]) + ", "  +  str(ghostPositions[0][1]) + ", " + \
                str(ghostPositions[1][0]) + ", "  +  str(ghostPositions[1][1]) + ", " + \
                str(ghostPositions[2][0]) + ", "  +  str(ghostPositions[2][1]) + ", " + \
                str(ghostPositions[3][0]) + ", "  +  str(ghostPositions[3][1]) + ", " + \
                str(legalNorth) + ", " + str(legalSouth) + ", " + \
                str(legalWest) + ", " + str(legalEast) + ", " + \
                str(ghostDirections.get(0))  + ", " + str(ghostDirections.get(1))  + ", " + \
                str(ghostDirections.get(2))  + ", " + str(ghostDirections.get(3))  + ", " + \
                str(ghost1Distance) + ", " + str(ghost2Distance) + ", " + \
                str(ghost3Distance) + ", " + str(ghost4Distance) + ", " + \
                str(livingGhost[1]) + ", "  +  str(livingGhost[2]) + ", " + \
                str(livingGhost[3]) + ", "  +  str(livingGhost[4]) + ", " + str(self.currentMove)
                