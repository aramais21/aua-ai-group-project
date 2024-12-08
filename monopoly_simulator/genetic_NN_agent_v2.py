import os
import copy
import random
import json
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import logging

from monopoly_simulator import background_agent_v3_1, mcmc_agent_based_on_landing_possibility, Hybrid_ANN_background, \
    initialize_game_elements
from monopoly_simulator.agent import Agent
from monopoly_simulator.mcmc_background_agent import _simulate_action

# Logger setup
logger = logging.getLogger('monopoly_simulator.logging_info.genetic_nn_agent')

# Example hyperparameter ranges for optimization
HYPERPARAMETER_RANGES = {
    "learning_rate": [1e-5, 1e-2],
    "hidden_layer_1": [64, 1024],
    "hidden_layer_2": [32, 512],
    "batch_size": [32, 256],
    "gamma": [0.9, 0.999],
}


def initialize_model(hyperparameters):
    """
    Initializes the neural network based on hyperparameters.
    """
    model = Sequential([
        Input(shape=(240,)),  # Input layer with state size (e.g., 240 for Monopoly)
        Dense(hyperparameters["hidden_layer_1"], activation='relu'),
        Dense(2922, activation='softmax')  # Output layer with action space size
    ])

    model.compile(
        optimizer=Adam(learning_rate=hyperparameters["learning_rate"]),
        loss='categorical_crossentropy',
    )
    return model


def encode_game_state(player, current_gameboard):
    """
    Encodes the game state as a vector for the neural network.
    """
    state = np.zeros(240)  # Example vector size, adjust as necessary

    # Example: Encode player's cash and position
    state[0] = player.current_cash
    state[1] = player.current_position

    # Encode property ownership
    for i, location in enumerate(current_gameboard['location_sequence']):
        # Skip locations that don't have an 'owner' attribute
        if hasattr(location, 'owner'):
            state[2 + i] = 1 if location.owner == player else 0

    return state


def genetic_algorithm_optimization(num_generations, population_size):
    """
    Genetic algorithm to optimize hyperparameters.
    """
    def generate_random_hyperparameters():
        return {
            "learning_rate": 10 ** random.uniform(np.log10(HYPERPARAMETER_RANGES["learning_rate"][0]),
                                                  np.log10(HYPERPARAMETER_RANGES["learning_rate"][1])),
            "hidden_layer_1": random.randint(HYPERPARAMETER_RANGES["hidden_layer_1"][0],
                                             HYPERPARAMETER_RANGES["hidden_layer_1"][1]),
            "batch_size": random.randint(HYPERPARAMETER_RANGES["batch_size"][0],
                                         HYPERPARAMETER_RANGES["batch_size"][1]),
            "gamma": random.uniform(HYPERPARAMETER_RANGES["gamma"][0], HYPERPARAMETER_RANGES["gamma"][1]),
        }

    def evaluate_hyperparameters(hyperparameters):
        """
        Evaluate hyperparameters by training the model for a few episodes and returning a fitness score.
        """
        model = initialize_model(hyperparameters)
        total_reward = 0

        # Run a few simulated games and compute average reward
        for _ in range(5):  # Simulate 5 games
            reward = run_simulation(model, hyperparameters)
            total_reward += reward

        return total_reward / 5  # Average reward

    def run_simulation(model, hyperparameters):
        """
        Simulates a Monopoly game and evaluates the performance of the genetic NN agent.
        Reward is based on the final cash of the genetic NN agent.
        """

        # Initialize player agents
        player_decision_agents = {
            'player_1': Agent(**background_agent_v3_1.decision_agent_methods),
            'player_2': Agent(**mcmc_agent_based_on_landing_possibility.decision_agent_methods),
            'player_3': Agent(**genetic_NN_agent.nn_decision_agent_methods),  # Genetic NN agent
            'player_4': Agent(**background_agent_v3_1.decision_agent_methods),
        }

        game_schema_file_path = '/Users/taron.schisas/Desktop/PycharmProjects/GNOME-p3/monopoly_game_schema_v1-2.json'
        game_schema = json.load(open(game_schema_file_path, 'r'))
        game_elements = initialize_game_elements.initialize_board(game_schema, player_decision_agents)

        # Simulate the performance of the genetic NN agent (player_3)
        player_3 = game_elements['players'][2]  # Assuming player_3 is at index 2
        simulated_gameboard = copy.deepcopy(game_elements)

        # Simulate actions for the genetic NN agent
        total_reward = 0
        for _ in range(50):  # Simulate 50 decision steps
            allowable_moves = ["buy_property", "skip_turn", "roll_dice", "end_turn"]
            action_probs = model.predict(encode_game_state(player_3, simulated_gameboard)[np.newaxis, :])[0]
            action_index = np.argmax(action_probs)
            chosen_action = allowable_moves[action_index] if action_index < len(allowable_moves) else "skip_turn"

            # Simulate the action
            try:
                reward = _simulate_action(
                    getattr(player_decision_agents['player_3'], chosen_action, None),
                    {},
                    player_3,
                    simulated_gameboard,
                    num_simulations=1,
                )
                total_reward += reward
            except Exception as e:
                logger.warning(f"Error during simulation: {e}")
                continue

        return total_reward

    # Initialize population
    population = [generate_random_hyperparameters() for _ in range(population_size)]

    for generation in range(num_generations):
        logger.info(f"Starting generation {generation + 1}")

        # Evaluate population
        fitness_scores = [evaluate_hyperparameters(individual) for individual in population]

        # Select top individuals
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda item: item[0], reverse=True)]
        top_individuals = sorted_population[:population_size // 2]

        # Crossover and mutation
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(top_individuals, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Return best individual from the final generation
    best_hyperparameters = max(population, key=lambda ind: evaluate_hyperparameters(ind))
    return best_hyperparameters


def crossover(parent1, parent2):
    """
    Combine two parents to create a new individual.
    """
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child


def mutate(individual):
    """
    Randomly mutate an individual's hyperparameters.
    """
    mutation_probability = 0.2
    for key in individual:
        if random.random() < mutation_probability:
            if key == "learning_rate":
                individual[key] = 10 ** random.uniform(np.log10(HYPERPARAMETER_RANGES[key][0]),
                                                       np.log10(HYPERPARAMETER_RANGES[key][1]))
            elif key in ["hidden_layer_1", "batch_size"]:
                individual[key] = random.randint(HYPERPARAMETER_RANGES[key][0], HYPERPARAMETER_RANGES[key][1])
            elif key == "gamma":
                individual[key] = random.uniform(HYPERPARAMETER_RANGES[key][0], HYPERPARAMETER_RANGES[key][1])
    return individual


# Example usage of the genetic algorithm
if __name__ == "__main__":
    best_hyperparameters = genetic_algorithm_optimization(num_generations=10, population_size=20)  # changed
    trained_model = initialize_model(best_hyperparameters)
    print(f"Best hyperparameters: {best_hyperparameters}")
    # Save the trained model
    trained_model.save('my_model_v3_2.keras', include_optimizer=False)


# Load the trained model
model_path = 'my_model_v3_2.keras'
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    trained_model = load_model(model_path)


# Functions for game decision-making
def make_pre_roll_move(player, current_gameboard, allowable_moves, code):
    return _nn_decision(player, current_gameboard, allowable_moves, code, phase="pre-roll")


def make_post_roll_move(player, current_gameboard, allowable_moves, code):
    return _nn_decision(player, current_gameboard, allowable_moves, code, phase="post-roll")


def make_out_of_turn_move(player, current_gameboard, allowable_moves, code):
    return _nn_decision(player, current_gameboard, allowable_moves, code, phase="out-of-turn")


def handle_negative_cash_balance(player, current_gameboard):
    allowable_moves = ["mortgage_property", "sell_house", "declare_bankruptcy"]
    return _nn_decision(player, current_gameboard, allowable_moves, None, phase="handle-negative-cash")


def make_buy_property_decision(player, current_gameboard, asset):
    allowable_moves = ["buy_property", "skip_turn"]
    return _nn_decision(player, current_gameboard, allowable_moves, None, phase="buy-property") == "buy_property"


def make_bid(player, current_gameboard, asset, current_bid):
    allowable_moves = ["bid", "pass"]
    return _nn_decision(player, current_gameboard, allowable_moves, None, phase="bidding") == "bid"


def _nn_decision(player, current_gameboard, allowable_moves, code, phase):
    """
    Uses the trained neural network to make a decision.
    """

    if not allowable_moves:
        logger.warning("No allowable moves provided!")
        return "skip_turn", {}
    logger.debug(f"Allowable moves received: {allowable_moves}")

    allowable_moves = list(allowable_moves)
    if not allowable_moves:
        logger.warning("No allowable moves available.")
        return None, {}

    # Encode the game state
    state = encode_game_state(player, current_gameboard)

    # Predict action probabilities
    action_probs = trained_model.predict(state[np.newaxis, :])[0]

    # Get the action index with the highest probability
    action_index = np.argmax(action_probs)

    # Map the action index to allowable moves
    if action_index < 0 or action_index >= len(allowable_moves):
        logger.warning("Invalid action_index derived. Defaulting to 'skip_turn'.")
        return "skip_turn", {}

    return allowable_moves[action_index]


# Agent methods dictionary
nn_decision_agent_methods = {
    "handle_negative_cash_balance": handle_negative_cash_balance,
    "make_pre_roll_move": make_pre_roll_move,
    "make_post_roll_move": make_post_roll_move,
    "make_out_of_turn_move": make_out_of_turn_move,
    "make_buy_property_decision": make_buy_property_decision,
    "make_bid": make_bid,
    "type": "decision_agent_methods",
}