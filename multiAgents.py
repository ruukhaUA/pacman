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

import torch
import numpy as np
from net import PacmanNet
import os
from util import manhattanDistance
from game import Directions
import random, util
random.seed(42)  # For reproducibility
from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

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
        "*** YOUR CODE HERE ***"

        best_score = float("-inf")
        best_action = Directions.STOP

        for action in gameState.getLegalActions(0):
            score = self.value(gameState.generateSuccessor(0, action), 1, 0)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action
    
    def value(self, state, agent, depth):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        actions = state.getLegalActions(agent)
        if not actions:
            return self.evaluationFunction(state)

        next_agent = (agent + 1) % state.getNumAgents()
        next_depth = depth + 1 if next_agent == 0 else depth

        scores = [self.value(state.generateSuccessor(agent, action), next_agent, next_depth) for action in actions]
        return max(scores) if agent == 0 else min(scores)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def value(self, state, agent, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        actions = state.getLegalActions(agent)
        if not actions:
            return self.evaluationFunction(state)

        next_agent = (agent + 1) % state.getNumAgents()
        next_depth = depth + 1 if next_agent == 0 else depth

        if agent == 0:
            best_score = float("-inf")
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                best_score = max(best_score, self.value(successor, next_agent, next_depth, alpha, beta))
                if best_score > beta:
                    return best_score
                alpha = max(alpha, best_score)
            return best_score
        else:
            best_score = float("inf")
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                best_score = min(best_score, self.value(successor, next_agent, next_depth, alpha, beta))
                if best_score < alpha:
                    return best_score
                beta = min(beta, best_score)
            return best_score
        
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        best_score = float("-inf")
        best_action = Directions.STOP
        alpha = float("-inf")
        beta = float("inf")
        current_direction = gameState.getPacmanState().configuration.direction

        for action in gameState.getLegalActions(0):
            score = self.value(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)

            if action == Directions.STOP:
                score -= 1000

            if action == Directions.REVERSE[current_direction]:
                score -= 50

            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, best_score)

        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def value(self, state, agent, depth):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        actions = state.getLegalActions(agent)
        if not actions:
            return self.evaluationFunction(state)

        next_agent = (agent + 1) % state.getNumAgents()
        next_depth = depth + 1 if next_agent == 0 else depth

        scores = [
            self.value(state.generateSuccessor(agent, action), next_agent, next_depth)
            for action in actions
        ]

        if agent == 0:
            return max(scores)
        else:
            return sum(scores) / len(scores)
        
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        best_score = float("-inf")
        best_action = Directions.STOP

        for action in gameState.getLegalActions(0):
            score = self.value(gameState.generateSuccessor(0, action), 1, 0)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    score = currentGameState.getScore()

    pacman_pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    walls = currentGameState.getWalls()

    x, y = pacman_pos

    # Factor 1: Distancia a la comida más cercana
    if food:
        min_food_distance = min(
            manhattanDistance(pacman_pos, food_pos)
            for food_pos in food
        )
        score += 10.0 / (min_food_distance + 1)

        # Penalizar que quede mucha comida
        score -= 10.0 * len(food)

    # Factor extra: preferir cápsulas si todavía quedan
    if capsules:
        min_capsule_distance = min(
            manhattanDistance(pacman_pos, capsule_pos)
            for capsule_pos in capsules
        )
        score += 8.0 / (min_capsule_distance + 1)
        score -= 15.0 * len(capsules)

    # Factor 2: Proximidad a fantasmas
    for ghost_state in ghost_states:
        ghost_pos = ghost_state.getPosition()
        ghost_distance = manhattanDistance(pacman_pos, ghost_pos)

        if ghost_state.scaredTimer > 0:
            # Si el fantasma está asustado, acercarse a él
            score += 50.0 / (ghost_distance + 1)
        else:
            # Si no está asustado, evitarlo
            if ghost_distance <= 1:
                score -= 500
            elif ghost_distance <= 2:
                score -= 200
            else:
                score -= 2.0 / ghost_distance

    # Factor 3: Cantidad de comida en cada dirección
    direction_vectors = {
        Directions.NORTH: (0, 1),
        Directions.SOUTH: (0, -1),
        Directions.EAST: (1, 0),
        Directions.WEST: (-1, 0)
    }

    legal_actions = currentGameState.getLegalActions(0)

    for action, (dx, dy) in direction_vectors.items():
        if action in legal_actions:
            food_in_direction = 0
            next_x, next_y = x + dx, y + dy

            while not walls[next_x][next_y]:
                if (next_x, next_y) in food:
                    food_in_direction += 1

                next_x += dx
                next_y += dy

            score += food_in_direction * 3

    # Factor 4: Número de paredes adyacentes
    adjacent_walls = 0

    for dx, dy in direction_vectors.values():
        next_x, next_y = x + dx, y + dy
        if walls[next_x][next_y]:
            adjacent_walls += 1

    score -= adjacent_walls * 5

    # Factor 5: Distancia a la comida solitaria
    if food:
        lonely_food = 0

        for food_pos in food:
            nearby_food = 0

            for other_food in food:
                if food_pos != other_food:
                    dist_between_food = manhattanDistance(food_pos, other_food)
                    if dist_between_food <= 3:
                        nearby_food += 1

            if nearby_food <= 1:
                dist_from_pacman = manhattanDistance(pacman_pos, food_pos)
                lonely_food += dist_from_pacman 

        score -= lonely_food * 3

    # Factor 4: acabar la comida restante
    if food and len(food) <= 10:
        closest_food = min(manhattanDistance(pacman_pos, f) for f in food)
        score += 300.0 / (closest_food + 1)
        score -= 30.0 * len(food)

    return score

# Abbreviation
better = betterEvaluationFunction


###########################################################################
# Ahmed
###########################################################################

class NeuralAgent(Agent):
    """
    Un agente de Pacman que utiliza una red neuronal para tomar decisiones
    basado en la evaluación del estado del juego.
    """
    def __init__(self, model_path="models/pacman_model.pth"):
        super().__init__()
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        
        # Mapeo de índices a acciones
        self.idx_to_action = {
            0: Directions.STOP,
            1: Directions.NORTH,
            2: Directions.SOUTH,
            3: Directions.EAST,
            4: Directions.WEST
        }
        
        # Para evaluar alternativas
        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}
        
        # Contador de movimientos
        self.move_count = 0
        
        print(f"NeuralAgent inicializado, usando dispositivo: {self.device}")

    def load_model(self, model_path):
        """Carga el modelo desde el archivo guardado"""
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: No se encontró el modelo en {model_path}")
                return False
                
            # Cargar el modelo
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_size = checkpoint['input_size']
            
            # Crear y cargar el modelo
            self.model = PacmanNet(self.input_size, 128, 5).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Modo evaluación
            
            print(f"Modelo cargado correctamente desde {model_path}")
            print(f"Tamaño de entrada: {self.input_size}")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False

    def state_to_matrix(self, state):
        """Convierte el estado del juego en una matriz numérica normalizada"""
        # Obtener dimensiones del tablero
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # Crear una matriz numérica
        # 0: pared, 1: espacio vacío, 2: comida, 3: cápsula, 4: fantasma, 5: Pacman
        numeric_map = np.zeros((width, height), dtype=np.float32)
        
        # Establecer espacios vacíos (todo lo que no es pared comienza como espacio vacío)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1
        
        # Agregar comida
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2
        
        # Agregar cápsulas
        for x, y in state.getCapsules():
            numeric_map[x][y] = 3
        
        # Agregar fantasmas
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            # Si el fantasma está asustado, marcarlo diferente
            if ghost_state.scaredTimer > 0:
                numeric_map[ghost_x][ghost_y] = 6  # Fantasma asustado
            else:
                numeric_map[ghost_x][ghost_y] = 4  # Fantasma normal
        
        # Agregar Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        numeric_map[int(pacman_x)][int(pacman_y)] = 5
        
        # Normalizar
        numeric_map = numeric_map / 6.0
        
        return numeric_map

    def evaluationFunction(self, state):
        """
        Una función de evaluación basada en la red neuronal y en heurísticas adicionales.
        """
        if self.model is None:
            return 0  # Si no hay modelo, devolver 0
        
        # Convertir a matriz
        state_matrix = self.state_to_matrix(state)
        
        # Convertir a tensor
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        # Obtener predicciones
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Aplicar heurísticas adicionales, similar a betterEvaluationFunction
        score = state.getScore()
        
        # Mejorar la evaluación con conocimiento del dominio
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()
        
        # Factor 1: Distancia a la comida más cercana
        if food:
            min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
            score += 1.0 / (min_food_distance + 1)
        
        # Factor 2: Proximidad a fantasmas
        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
            
            if ghost_state.scaredTimer > 0:
                # Si el fantasma está asustado, acercarse a él
                score += 50 / (ghost_distance + 1)
            else:
                # Si no está asustado, evitarlo
                if ghost_distance <= 2:
                    score -= 200  # Gran penalización por estar demasiado cerca

        # Factor 3: Cantidad de comida en esa dirección
        walls = state.getWalls()
        x, y = pacman_pos

        direction_vectors = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0)
        }

        for action, (dx, dy) in direction_vectors.items():
            if action in legal_actions:
                food_in_direction = 0
                next_x, next_y = x + dx, y + dy

                while not walls[next_x][next_y]:
                    if (next_x, next_y) in food:
                        food_in_direction += 1

                    next_x += dx
                    next_y += dy

                action_index = list(self.idx_to_action.values()).index(action)
                score += probabilities[action_index] * food_in_direction * 5

        # Factor 4: Número de paredes adyacentes
        adjacent_walls = 0

        for dx, dy in direction_vectors.values():
            next_x, next_y = x + dx, y + dy
            if walls[next_x][next_y]:
                adjacent_walls += 1

        score -= adjacent_walls * 10
        
        # Combinar la puntuación de la red con la heurística
        neural_score = 0
        for i, action in enumerate(self.idx_to_action.values()):
            if action in legal_actions:
                neural_score += probabilities[i] * 100
        
        return score + neural_score

    def getAction(self, state):
        """
        Devuelve la mejor acción basada en la evaluación de la red neuronal
        y heurísticas adicionales.
        """
        self.move_count += 1
        
        # Si no hay modelo, hacer un movimiento aleatorio
        if self.model is None:
            print("ERROR: Modelo no cargado. Haciendo movimiento aleatorio.")
            exit()
            legal_actions = state.getLegalActions()
            return random.choice(legal_actions)
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Evaluación directa con la red neuronal
        state_matrix = self.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Mapear índices del modelo a acciones del juego
        action_probs = []
        for idx, prob in enumerate(probabilities):
            action = self.idx_to_action[idx]
            if action in legal_actions:
                action_probs.append((action, prob))
        
        # Ordenar por probabilidad (mayor a menor)
        action_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Exploración: con una probabilidad decreciente, elegir aleatoriamente
        exploration_rate = 0.2 * (0.99 ** self.move_count)  # Disminuye con el tiempo
        if random.random() < exploration_rate:
            # Excluir STOP si es posible
            if len(legal_actions) > 1 and Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            return random.choice(legal_actions)
        
        # Evaluación alternativa: generar sucesores y evaluar cada uno
        successors = []
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            eval_score = self.evaluationFunction(successor)
            neural_score = 0
            for a, p in action_probs:
                if a == action:
                    neural_score = p * 100
                    break
            # Combinar evaluación heurística con la predicción de la red
            combined_score = eval_score + neural_score
            
            # Penalizar STOP a menos que sea la única opción
            if action == Directions.STOP and len(legal_actions) > 1:
                combined_score -= 50
                
            successors.append((action, combined_score))
        
        # Ordenar por puntuación combinada
        successors.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver la mejor acción
        return successors[0][0]

# Definir una función para crear el agente
def createNeuralAgent(model_path="models/pacman_model.pth"):
    """
    Función de fábrica para crear un agente neuronal.
    Útil para integrarse con la estructura de pacman.py.
    """
    return NeuralAgent(model_path)

class AlphaBetaNeuralAgent(AlphaBetaAgent):
    def __init__(self, depth='2', model_path="models/pacman_model.pth", decay = 0.01, reverse_decay = False):
        super().__init__('scoreEvaluationFunction', depth)
        self.model = createNeuralAgent(model_path)
        self.decay = float(decay)
        self.reverse_decay = bool(reverse_decay)
        self.initial_food_count = None

        self.evaluationFunction = self.evaluation_combined

    def neural_evaluation(self, state):
        if self.model is None or self.model.model is None:
            return 0  # Si no hay modelo, devolver 0
        
        # Convertir a matriz
        state_matrix = self.model.state_to_matrix(state)
        
        # Convertir a tensor
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.model.device)
        
        # Obtener predicciones
        with torch.no_grad():
            output = self.model.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

        legal_actions = state.getLegalActions()
        neural_score = 0
        for i, action in enumerate(self.model.idx_to_action.values()):
            if action in legal_actions:
                neural_score += probabilities[i] * 100

        return neural_score

    def evaluation_combined(self, state):
        food = state.getFood().asList()

        # Save initial amount once
        if self.initial_food_count is None:
            self.initial_food_count = len(food)

        food_ratio = len(food) / self.initial_food_count

        w_neural = food_ratio
        w_heuristic = 1 - w_neural

        # 1) Traditional score (with the new heuristics from Task 1)
        trad_score = betterEvaluationFunction(state)
        
        # 2) Neural network score
        neural_score = self.neural_evaluation(state)
        
        # 3) Weighted combination
        return w_heuristic * trad_score + w_neural * neural_score