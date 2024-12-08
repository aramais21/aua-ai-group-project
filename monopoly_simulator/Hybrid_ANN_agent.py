import os
from tensorflow.keras.models import load_model

from monopoly_simulator import genetic_NN_agent_v2
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
    return genetic_NN_agent_v2._nn_decision(player, current_gameboard, allowable_moves, code, phase="pre-roll")


def make_post_roll_move(player, current_gameboard, allowable_moves, code):
    return genetic_NN_agent_v2._nn_decision(player, current_gameboard, allowable_moves, code, phase="post-roll")


def make_out_of_turn_move(player, current_gameboard, allowable_moves, code):
    return genetic_NN_agent_v2._nn_decision(player, current_gameboard, allowable_moves, code, phase="out-of-turn")


def handle_negative_cash_balance(player, current_gameboard):
    allowable_moves = ["mortgage_property", "sell_house", "declare_bankruptcy"]
    return genetic_NN_agent_v2._nn_decision(player, current_gameboard, allowable_moves, None, phase="handle-negative-cash")


def make_buy_property_decision(player, current_gameboard, asset):
    # Use background agent for property purchase decisions
    return background_agent["make_buy_property_decision"](player, current_gameboard, asset)


def make_bid(player, current_gameboard, asset, current_bid):
    # Use background agent for bidding
    return background_agent["make_bid"](player, current_gameboard, asset, current_bid)


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
