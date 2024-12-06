import numpy as np
import os
from tensorflow.keras.models import load_model
from monopoly_simulator.background_agent_v3_1 import decision_agent_methods as background_agent
import logging
# fix action index, game state


# Logger setup
logger = logging.getLogger('monopoly_simulator.logging_info.hybrid_agent')

# Load the trained model for the ANN agent
model_path = 'my_model_v3.keras'
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    trained_model = load_model(model_path)


# Hybrid agent combining background agent for basic decisions and ANN agent for complex decisions
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
    # Use background agent for property purchase decisions
    return background_agent["make_buy_property_decision"](player, current_gameboard, asset)


def make_bid(player, current_gameboard, asset, current_bid):
    # Use background agent for bidding
    return background_agent["make_bid"](player, current_gameboard, asset, current_bid)


def _nn_decision(player, current_gameboard, allowable_moves, code, phase):
    """
    Uses the trained neural network to make a decision.
    """
    allowable_moves = list(allowable_moves)
    if not allowable_moves:
        logger.warning("No allowable moves available.")
        return None

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

    return allowable_moves[action_index], {}


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



# Hybrid agent methods dictionary
hybrid_decision_agent_methods = {
    "handle_negative_cash_balance": handle_negative_cash_balance,
    "make_pre_roll_move": make_pre_roll_move,
    "make_post_roll_move": make_post_roll_move,
    "make_out_of_turn_move": make_out_of_turn_move,
    "make_buy_property_decision": make_buy_property_decision,
    "make_bid": make_bid,
    "type": "decision_agent_methods",
}
