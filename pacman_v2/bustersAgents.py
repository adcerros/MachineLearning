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
import copy
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
from os.path import exists

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

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

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
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
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
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move

    def printLineData(self, gameState):
        return "XXXXXXXXXX"


class QLearningAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.maxDistance = gameState.data.layout.width + gameState.data.layout.height - 2
        self.maxAgents = 3
        self.initQtable()
        self.table_file = open('qtable.txt', 'r+')
#        self.table_file_csv = open("qtable.csv", "r+")        
        self.q_table = self.readQtable()
        self.epsilon = 0.1
        self.alpha = 0.2
        self.discount = 0.9
        self.countActions = 0
        self.score = 0
        self.scoreDiff = 0
        self.positionsList = []


    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable


    def getNearestFoodDistanceAndPos(self):
        if(self.gameState.getNumFood() > 0):
            minDistance = 900000
            pacmanPosition = self.gameState.getPacmanPosition()
            for i in range(self.gameState.data.layout.width):
                for j in range(self.gameState.data.layout.height):
                    if self.gameState.hasFood(i, j):
                        foodPosition = i, j
                        distance = util.manhattanDistance(pacmanPosition, foodPosition)
                        if distance < minDistance:
                            minDistance = distance
                            nearestFoodPos = (i,j)
            return minDistance, nearestFoodPos

        else:
            return None, None;

    # Establece el objetivo mas cercano a la posicion del pacman en el momento de la llamada
    def getNearlyObjectivePos(self, state):
        minFoodDistance, nearestFoodPos = self.getNearestFoodDistanceAndPos()
        # nealyGhost = [manhattanDistance, ghostNumber]
        nearlyGhost = [1000, 0]
        livingGhosts = state.getLivingGhosts()
        for i in range(len(state.data.ghostDistances)):
            if livingGhosts[i + 1] and state.data.ghostDistances[i] != None :
                if state.data.ghostDistances[i] < nearlyGhost[0]:
                    nearlyGhost = [state.data.ghostDistances[i], i]
        self.currentNearlyGhostIndex = nearlyGhost[1]
        self.nearlyGhostDistance = nearlyGhost[0]
        if minFoodDistance != None and nearestFoodPos != None:
            if self.nearlyGhostDistance > minFoodDistance:
                return nearestFoodPos
        ghostsPositions = state.getGhostPositions()
        return list(ghostsPositions[self.currentNearlyGhostIndex])


    def initQtable(self):
        if not exists('./qtable.txt'):
            self.table_file = open('qtable.txt', 'w+') 
            # Numero de posiciones realtivas * distancia maxima * numero maximo fantasmas * numero de estados de los muros (no se puede alcanzar el estado rodeado por muros en las 4 direcciones) * 4 (direccion Pacman)
            for i in range(8 * 6 * 15 * 4):
                for j in range(4):
                    self.table_file.write(str(0)+" ")
                self.table_file.write("\n")


    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()
        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

            
    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")    
            
    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()


    # state = (relativePosition, distance, wallsState, Pacman_direction)
    def computePosition(self, state):
        return (state[0] * (6 * 15 * 4)) + (state[1] * 15 * 4) + (state[2] * 4) + state[3]


    def getQValue(self, state, action):

        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.getActionColumn(action)
        return self.q_table[position][action_column]

    def getActionColumn(self, action):
        if action == Directions.NORTH:
            return 0
        elif action == Directions.SOUTH:
            return 1
        elif action == Directions.EAST:
            return 2
        elif action == Directions.WEST:
            return 3
        else: 
            return 0



    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.legalActions
        if len(legalActions)==0:
          return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        action = None
        self.legalActions = state.getLegalPacmanActions()
        self.legalActions = self.legalActions[0: len(self.legalActions) - 1]
        if len(self.legalActions) == 0:
             return action
        flip = util.flipCoin(self.epsilon)
        if flip:
            action = random.choice(self.legalActions)
            return action
        self.updateGameStateInfo(state)
        action, q_currentState, q_nextState = self.createNewStates(state)
        self.update(q_currentState, action, q_nextState)
        return action

    def createNewStates(self, state):
        nearlyGhostPos = self.getNearlyObjectivePos(state)
        pacmanPosition = list(state.getPacmanPosition())
        walls = state.getWalls()
        q_currentState = self.calculateMyCurrentState(nearlyGhostPos, pacmanPosition, walls)
        action = self.getPolicy(q_currentState)
        q_nextState = self.calculateMyNextState(action, nearlyGhostPos, pacmanPosition, walls)
        return action,q_currentState,q_nextState

    def updateGameStateInfo(self, state):
        self.gameState = state
        self.scoreDiff = self.gameState.getScore() - self.score
        self.score = self.gameState.getScore() 
        self.countActions += 1


    def calculateStateOfWalls(self, pacmanPosition, walls):
        # estado de los muros binario de 4 bits cada bit indica 0 no hay muro 1 hay muro
        # Ej: 0001 = muro norte, sur, este = false ;  muro oeste = true;
        stateOfWalls = 0
        if walls[pacmanPosition[0] + 1][pacmanPosition[1]] == True:
            stateOfWalls += 8
        if walls[pacmanPosition[0] - 1][pacmanPosition[1]] == True:
            stateOfWalls += 4
        if walls[pacmanPosition[0]][pacmanPosition[1] + 1] == True:
            stateOfWalls += 2
        if walls[pacmanPosition[0]][pacmanPosition[1] + 1] == True:
            stateOfWalls += 1
        return stateOfWalls

    # Construccion del estado actual en forma de tupla  
    def calculateMyCurrentState(self, nearlyGhostPos, pacmanPosition, walls):
        distance = self.getNearlyGhostDistance(pacmanPosition, nearlyGhostPos)
        relativePosition = self.getRelativePosition(pacmanPosition, nearlyGhostPos)
        stateOfWalls = self.calculateStateOfWalls(pacmanPosition, walls)
        direction = self.getActionColumn(self.gameState.data.agentStates[0].getDirection())
        self.updatePositionsList(pacmanPosition)
        q_currentState = (relativePosition, distance, stateOfWalls, direction)
        return q_currentState

    # Retorna: si esta repetido mas de 2 veces retorna 3, sino retorna el numero de veces repetido
    def calculateDirection(self, pacmanPosition):
        timesRepeated = self.positionsList.count(pacmanPosition)
        if timesRepeated > 2:
            return 3
        return timesRepeated

    # Construccion del siguiente estado en forma de tupla
    def calculateMyNextState(self, action, nearlyGhostPos, pacmanPosition, walls):
        self.next_pacmanPosition = self.getNextPosition(action, pacmanPosition)
        next_distance = self.getNearlyGhostDistance(self.next_pacmanPosition, nearlyGhostPos)
        next_relativePosition = self.getRelativePosition(pacmanPosition, nearlyGhostPos)
        stateOfWalls = self.calculateStateOfWalls(self.next_pacmanPosition, walls)
        direction = self.getActionColumn(action)
        q_nextState = (next_relativePosition, next_distance, stateOfWalls, direction)
        return q_nextState

    def updatePositionsList(self, next_pacmanPosition):
        if len(self.positionsList) > 1000:
            self.positionsList.pop(0)
        self.positionsList.append(next_pacmanPosition)
    

    def getNextPosition(self, action, position):
        next_pacmanPosition = copy.deepcopy(position)
        if action == "North":
            next_pacmanPosition[1] = next_pacmanPosition[1] + 1
        elif action == "South":
            next_pacmanPosition[1] = next_pacmanPosition[1] - 1
        elif action == "East":
            next_pacmanPosition[0] = next_pacmanPosition[0] + 1
        elif action == "West":
            next_pacmanPosition[0] = next_pacmanPosition[0] - 1
        elif action == "Stop":
            return next_pacmanPosition
        return next_pacmanPosition


    def getNearlyGhostDistance(self, position, nearlyGhostPos):
        diferenceX =  abs(nearlyGhostPos[0] - position[0])
        diferenceY =  abs(nearlyGhostPos[1] - position[1])
        distance = diferenceX + diferenceY
        # Se discretiza la distancia
        if distance <= 1:
            return 0  # Muy cerca
        elif distance > 1 and distance <=2:
            return 1  # Cerca
        elif distance > 2 and distance <=5:
            return 2  # Media
        elif distance > 5 and distance <=10:
            return 3  # Lejos      
        elif distance > 10 and distance <=20:
            return 4  # Muy lejos
        else:
            return 5 # Extremadamente lejos


    def getRelativePosition(self, position, nearlyGhostPos): 
        diferenceX =  nearlyGhostPos[0] - position[0]
        diferenceY =  nearlyGhostPos[1] - position[1]
        if diferenceX < 0:
            if diferenceY < 0:
                return 0 #SouthWest
            elif diferenceY > 0:
                return 1 #NorthWest
            elif diferenceY == 0:
                return 2 #West
        elif diferenceX == 0:
            if diferenceY < 0:
                return 3 #South
            elif diferenceY > 0:
                return 4 #North
        elif diferenceX > 0:    
            if diferenceY < 0:
                return 5 #SouthEast
            elif diferenceY > 0:
                return 6 #NorthEast
            elif diferenceY == 0:
                return 7 #East

    def update(self, state, action, nextState):
        action_column = self.getActionColumn(action)
        position = self.computePosition(state)
        self.reward = self.calculateReward(state, action, nextState)
        self.q_table[position][action_column] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (self.reward + self.discount * self.getQValue(nextState, self.getPolicy(nextState)))
        
        # TRACE for updated q-table. Comment the following lines if you do not want to see that trace
        # print("Q-table:")
        #self.printQtable()

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)


    def calculateReward(self, state, action, nextState):
        reward = 0
        if nextState[1] == 0:
            reward += 5
        elif nextState[1] == 1:
            reward += 2
        elif nextState[1] == 2:
            reward += 1
        elif nextState[1] == 3:
            reward += 0.5
        elif nextState[1] == 4:
            reward -= 1
        elif nextState[1] == 5:
            reward -= 2
        # pacmanPosition = list(self.gameState.getPacmanPosition())
        if self.next_pacmanPosition == self.getNearlyObjectivePos(self.gameState):
            reward += 200
        if self.gameState.hasFood(self.next_pacmanPosition[0], self.next_pacmanPosition[1]):
            reward += 400
        if self.positionsList.count(self.next_pacmanPosition) > 2:
            reward -= 0.5 * self.positionsList.count(self.next_pacmanPosition) - 1
        # print("Resultado: " + str(self.positionsList.count(self.next_pacmanPosition)) + "\n")
        return reward

