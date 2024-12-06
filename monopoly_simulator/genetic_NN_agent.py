import os
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import logging
from monopoly_simulator.flag_config import flag_config_dict

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
        Dense(hyperparameters["hidden_layer_2"], activation='relu'),
        Dense(2922, activation='softmax')  # Output layer with action space size
    ])

    model.compile(
        optimizer=Adam(learning_rate=hyperparameters["learning_rate"]),
        loss='categorical_crossentropy',
    )
    return model


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
            "hidden_layer_2": random.randint(HYPERPARAMETER_RANGES["hidden_layer_2"][0],
                                             HYPERPARAMETER_RANGES["hidden_layer_2"][1]),
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

        # Run a few games with random initialization and compute average reward
        for _ in range(5):  # Simulate 5 games
            reward = run_simulation(model, hyperparameters)
            total_reward += reward

        return total_reward / 5  # Average reward

    def run_simulation(model, hyperparameters):
        """
        Simulates a Monopoly game and evaluates the performance of the model.
        Reward is based on the total assets owned by all players.
        """
        total_rewards = []
        for player_id in range(4):  # Simulate for 4 players
            state = encode_game_state(player_id)
            action_probs = model.predict(state[np.newaxis, :])[0]
            action = np.argmax(action_probs)  # Choose the action with the highest probability

            # Simulate game outcomes
            simulated_reward = simulate_game_step(action, player_id)
            total_rewards.append(simulated_reward)

        return sum(total_rewards)

    def run_simulation(model, hyperparameters):
        """
        Simulates a single Monopoly game and evaluates the performance of the model.
        """
        total_reward = 0
        state = np.random.rand(240)  # Example: random state initialization

        for step in range(25):  # Simulate 50 decision steps
            action_probs = model.predict(state[np.newaxis, :])[0]
            action = np.argmax(action_probs)
            reward = np.random.uniform(-1, 1)  # Random reward for simulation purposes
            total_reward += reward

        return total_reward

    # Initialize population
    population = [generate_random_hyperparameters() for _ in range(population_size)]

    for generation in range(num_generations):
        logger.info(f"Starting generation {generation + 1}")

        # Evaluate population
        fitness_scores = [evaluate_hyperparameters(individual) for individual in population]

        # Select top individuals
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
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
            elif key in ["hidden_layer_1", "hidden_layer_2", "batch_size"]:
                individual[key] = random.randint(HYPERPARAMETER_RANGES[key][0], HYPERPARAMETER_RANGES[key][1])
            elif key == "gamma":
                individual[key] = random.uniform(HYPERPARAMETER_RANGES[key][0], HYPERPARAMETER_RANGES[key][1])
    return individual


# Example usage of the genetic algorithm
if __name__ == "__main__":
    best_hyperparameters = genetic_algorithm_optimization(num_generations=10, population_size=20) #changed
    trained_model = initialize_model(best_hyperparameters)
    print(f"Best hyperparameters: {best_hyperparameters}")
    # Save the trained model
    trained_model.save('my_model.keras', include_optimizer=False)
    # Best hyperparameters: {'learning_rate': 0.004949724177997446, 'hidden_layer_1': 649,
    # 'hidden_layer_2': 321, 'batch_size': 136, 'gamma': 0.9690224859894329} num_gen=5,pop_size=10
    
    # Best hyperparameters: {'learning_rate': 2.0884005280479127e-05, 'hidden_layer_1': 976, 'hidden_layer_2': 391,
    # 'batch_size': 241, 'gamma': 0.9280131743259252} num_generations=10, population_size=20


# Load the trained model
model_path = 'my_model.keras'
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

    # Additional encodings can be added here
    return state


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
