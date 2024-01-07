# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import numpy as np
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):
    def alphabeta(self, agent, depth, gameState, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return evaluationFunction(gameState)
        if agent == 0:  # maximize for pacman
            value = float("-inf")
            for action in getLegalActionsNoStop(agent, gameState):
                value = max(value,
                            self.alphabeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:  # alpha-beta pruning
                    break
            return value
        else:  # minimize for ghosts
            nextAgent = agent + 1  # get the next agent
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:  # increase depth every time all agents have moved
                depth += 1
            for action in getLegalActionsNoStop(agent, gameState):
                value = float("inf")
                value = min(value,
                            self.alphabeta(nextAgent, depth, gameState.generateSuccessor(agent, action), alpha,
                                           beta))
                beta = min(beta, value)
                if beta <= alpha:  # alpha-beta pruning
                    break
            return value

    def getAction(self, gameState: GameState):
        """
        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # TODO: Your code goes here
        # util.raiseNotDefined()
        possibleActions = getLegalActionsNoStop(0, gameState)
        # Initial state: agent_index = 0, depth = 0, alpha = -infinity, beta = +infinity
        alpha = float("-inf")
        beta = float("inf")
        action_scores = [self.alphabeta(0, 0, gameState.generateSuccessor(0, action), alpha, beta) for action
                         in possibleActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]

def evaluationFunction(currentGameState):

    # Setup information to be used as arguments in evaluation function
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()

    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)
    capsule_count = len(currentGameState.getCapsules())
    closest_food = 1

    game_score = currentGameState.getScore()

    # Find distances from pacman to all food
    food_distances = [manhattanDistance(pacman_position, food_position) for food_position in food_list]

    # Set value for closest food if there is still food left
    if food_count > 0:
        closest_food = min(food_distances)

    # Find distances from pacman to ghost(s)
    for ghost_position in ghost_positions:
        ghost_distance = manhattanDistance(pacman_position, ghost_position)

        # If ghost is too close to pacman, prioritize escaping instead of eating the closest food
        # by resetting the value for closest distance to food
        if ghost_distance < 2:
            closest_food = float("inf")

    features = [1.0 / closest_food,
                game_score,
                food_count,
                capsule_count]
    weights = [10,
               200,
               -100,
               -10]

    # Linear combination of features
    return sum([feature * weight for feature, weight in zip(features, weights)])

def getLegalActionsNoStop(index, gameState):
        possibleActions = gameState.getLegalActions(index)
        if Directions.STOP in possibleActions:
            possibleActions.remove(Directions.STOP)
        return possibleActions

