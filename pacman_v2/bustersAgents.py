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
        self.initQtable(gameState)
        self.table_file = open('qtable.txt', 'r+')
#        self.table_file_csv = open("qtable.csv", "r+")        
        self.q_table = self.readQtable()
        self.epsilon = 0.05
        self.alpha = 0.2
        self.discount = 0.8
        self.reward = 0
        self.countActions = 0


    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable


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
        self.nearlyGhostDistance = nearlyGhost[0]
        ghostsPositions = gameState.getGhostPositions()
        return ghostsPositions[self.currentNearlyGhostIndex]


    def initQtable(self, gameState):
        if not exists('./qtable.txt'):
            self.table_file = open('qtable.txt', 'w+') 
            for i in range(8 * self.maxDistance):
                for j in range(5):
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

#         self.table_file_csv.seek(0)
#         self.table_file_csv.truncate()
#         for line in self.q_table:
#             for item in line[:-1]:
#                 self.table_file_csv.write(str(item)+", ")
#             self.table_file_csv.write(str(line[-1]))                
#             self.table_file_csv.write("\n")

            
    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")    
            
    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()


    def computePosition(self, state):
        current_pacman_pos = state.getPacmanPosition()
        nearlyGhostPos = self.getNearlyGhostPos(state)
        diferenceX =  nearlyGhostPos[0] - current_pacman_pos[0]
        diferenceY =  nearlyGhostPos[1] - current_pacman_pos[1]
        if diferenceX < 0 and diferenceY < 0:
            return 0 * 8 + self.nearlyGhostDistance 
        elif diferenceX < 0 and diferenceY > 0:
            return 1 * 8 + self.nearlyGhostDistance 
        elif diferenceX < 0 and diferenceY == 0:
            return 2 * 8 + self.nearlyGhostDistance 
        elif diferenceX == 0 and diferenceY < 0:
            return 3 * 8 + self.nearlyGhostDistance 
        elif diferenceX == 0 and diferenceY > 0:
            return 4 * 8 + self.nearlyGhostDistance 
        elif diferenceX > 0 and diferenceY < 0:
            return 5 * 8 + self.nearlyGhostDistance 
        elif diferenceX > 0 and diferenceY > 0:
            return 6 * 8 + self.nearlyGhostDistance 
        elif diferenceX > 0 and diferenceY == 0:
            return 7 * 8 + self.nearlyGhostDistance 

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
        elif action == Directions.STOP:
            return 4

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = state.getLegalActions()
        if len(legalActions)==0:
          return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = state.getLegalPacmanActions()
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
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = state.getLegalPacmanActions()
        self.action = None
        self.countActions += 1
        if len(legalActions) == 0:
             return self.action
        flip = util.flipCoin(self.epsilon)
        if flip:
            self.action = random.choice(legalActions)
            return self.action
        self.action = self.getPolicy(state)
        return self.action



    def update(self, state, action, nextState):
        action_column = self.getActionColumn(action)
        position = self.computePosition(state)
        self.reward = self.calculateReward(state, action, nextState)
        self.q_table[position][action_column] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (self.reward + self.discount * self.getQValue(nextState, self.getPolicy(nextState)))
        
        # TRACE for updated q-table. Comment the following lines if you do not want to see that trace
        # print("Q-table:")
        self.printQtable()

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

    def calculateReward(self, state, action, nextState):
        if action == "Stop":
            return nextState.getScore() - state.getScore() - self.nearlyGhostDistance - 1000
        return nextState.getScore() - state.getScore() - self.nearlyGhostDistance
