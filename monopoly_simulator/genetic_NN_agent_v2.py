import os
import copy
import random
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import logging
from skopt import forest_minimize
from skopt.space import Real, Integer

from monopoly_simulator import background_agent_v3_1, mcmc_background_agent, Hybrid_ANN_background, \
    initialize_game_elements, genetic_NN_agent
from monopoly_simulator.agent import Agent
from monopoly_simulator.mcmc_background_agent import _simulate_action

# Logger setup
logger = logging.getLogger('monopoly_simulator.logging_info.genetic_nn_agent')

# Define the search space for hyperparameters
search_space = [
    Real(1e-5, 1e-2, name='learning_rate'),  # Learning rate
    Integer(64, 1024, name='hidden_layer_1'),  # Number of neurons in the first hidden layer
    Integer(32, 256, name='batch_size'),  # Batch size
    Real(0.9, 0.999, name='gamma')  # Discount factor (gamma)
]


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
        if hasattr(location, 'owner'):
            state[2 + i] = 1 if location.owner == player else 0

    return state


def evaluate_model(hyperparameters):
    """
    Evaluate the model with the given hyperparameters by running simulations.
    """
    learning_rate, hidden_layer_1, batch_size, gamma = hyperparameters

    # Construct a dictionary for the hyperparameters
    hyperparameters_dict = {
        "learning_rate": learning_rate,
        "hidden_layer_1": hidden_layer_1,
        "batch_size": batch_size,
        "gamma": gamma
    }

    # Initialize the model
    model = initialize_model(hyperparameters_dict)
    total_reward = 0

    # Run a few simulated games and compute average reward
    for _ in range(5):  # Simulate 5 games
        reward = run_simulation(model, hyperparameters_dict)
        total_reward += reward

    return -total_reward / 5  # Return negative for minimization



def run_simulation(model, hyperparameters):
    """
    Simulates a Monopoly game and evaluates the performance of the genetic NN agent.
    """
    import json
    game_schema_file_path = '/Users/taron.schisas/Desktop/PycharmProjects/GNOME-p3/monopoly_game_schema_v1-2.json'
    game_schema = json.load(open(game_schema_file_path, 'r'))
    player_decision_agents = {
        'player_1': Agent(**background_agent_v3_1.decision_agent_methods),
        'player_2': Agent(**mcmc_background_agent.decision_agent_methods),
        'player_3': Agent(**genetic_NN_agent.nn_decision_agent_methods),  # Genetic NN agent
        'player_4': Agent(**background_agent_v3_1.decision_agent_methods),
    }
    game_elements = initialize_game_elements.initialize_board(game_schema, player_decision_agents)

    player_3 = game_elements['players'][2]  # Assuming player_3 is at index 2
    simulated_gameboard = copy.deepcopy(game_elements)

    total_reward = 0
    for _ in range(50):  # Simulate 50 decision steps
        allowable_moves = ["buy_property", "skip_turn", "roll_dice", "end_turn"]  # Example
        action_probs = model.predict(encode_game_state(player_3, simulated_gameboard)[np.newaxis, :])[0]
        action_index = np.argmax(action_probs)
        chosen_action = allowable_moves[action_index] if action_index < len(allowable_moves) else "skip_turn"

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


# Initialize and train the model with the best hyperparameters
if __name__ == "__main__":
    # Optimize hyperparameters using Scikit-Optimize
    result = forest_minimize(
        func=evaluate_model,
        dimensions=search_space,
        acq_func='EI',  # Expected Improvement
        n_calls=50,  # Number of function evaluations
        random_state=42
    )

    # Print the best hyperparameters found
    print("Best hyperparameters:")
    print(result.x)
    # Best hyperparameters: {'learning_rate': 0.007967464438733727, 'hidden_layer_1': 334, 'batch_size': 138,
    #                   'gamma': 0.9771894090270041}

    # Initialize and train the model with the best hyperparameters
    best_hyperparameters = {
        "learning_rate": result.x[0],
        "hidden_layer_1": result.x[1],
        "batch_size": result.x[2],
        "gamma": result.x[3]
    }
    trained_model = initialize_model(best_hyperparameters)
    print(f"Best hyperparameters: {best_hyperparameters}")
    # Save the trained model
    trained_model.save('my_model_v3.keras', include_optimizer=False)


# Load the trained model
model_path = 'my_model_v3.keras'
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
    # Remaining logic...

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
