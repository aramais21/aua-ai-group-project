from monopoly_simulator.background_agent_v3_1 import decision_agent_methods as background_methods
from collections import Counter
import random

def make_pre_roll_move(player, current_gameboard, allowable_moves, code):
    """
    Decide the best pre-roll action. Uses background agent's logic for pre-roll moves.
    """
    return background_methods["make_pre_roll_move"](player, current_gameboard, allowable_moves, code)


def make_post_roll_move(player, current_gameboard, allowable_moves, code):
    """
    Decide the best post-roll action, integrating MCMC-based logic for property buying.
    """
    if "buy_property" in allowable_moves:
        current_location = current_gameboard['location_sequence'][player.current_position]
        if make_buy_property_decision(player, current_gameboard, current_location):
            return "buy_property", {
                "player": player.player_name,
                "asset": current_location.name,
                "current_gameboard": "current_gameboard",
            }
    return background_methods["make_post_roll_move"](player, current_gameboard, allowable_moves, code)


def make_out_of_turn_move(player, current_gameboard, allowable_moves, code):
    """
    Decide the best out-of-turn action. Falls back to the background agent's logic.
    """
    return background_methods["make_out_of_turn_move"](player, current_gameboard, allowable_moves, code)


def handle_negative_cash_balance(player, current_gameboard):
    """
    Handle negative cash balance using the background agent's logic.
    """
    return background_methods["handle_negative_cash_balance"](player, current_gameboard)


def make_buy_property_decision(player, current_gameboard, asset):
    """
    Decide whether to buy a property using landing probabilities.
    """
    landing_probs = _calculate_landing_probabilities(player, current_gameboard, num_simulations=2000)
    property_value = landing_probs.get(asset.name, 0) * 100
    return player.current_cash >= asset.price and property_value > asset.price


def make_bid(player, current_gameboard, asset, current_bid):
    """
    Decide the amount to bid for a property. Falls back to the background agent's logic.
    """
    return background_methods["make_bid"](player, current_gameboard, asset, current_bid)


def _calculate_landing_probabilities(player, current_gameboard, num_simulations=500):
    """
    Calculate landing probabilities for each square using Monte Carlo simulations.
    """
    position_counts = Counter()
    num_squares = len(current_gameboard['location_sequence'])

    for _ in range(num_simulations):
        current_position = player.current_position
        for _ in range(10):  # Simulate the next 10 moves
            dice_roll = random.randint(1, 6) + random.randint(1, 6)
            current_position = (current_position + dice_roll) % num_squares
            position_counts[current_position] += 1

    total_moves = sum(position_counts.values())
    return {
        current_gameboard['location_sequence'][pos].name: count / total_moves
        for pos, count in position_counts.items()
    }


def _build_decision_agent_methods_dict():
    """
    Build the decision agent methods dictionary for seamless integration.
    """
    return {
        "make_pre_roll_move": make_pre_roll_move,
        "make_post_roll_move": make_post_roll_move,
        "make_out_of_turn_move": make_out_of_turn_move,
        "handle_negative_cash_balance": handle_negative_cash_balance,
        "make_buy_property_decision": make_buy_property_decision,
        "make_bid": make_bid,
        "type": "decision_agent_methods",
    }


decision_agent_methods = _build_decision_agent_methods_dict()
