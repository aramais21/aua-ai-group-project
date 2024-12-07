from monopoly_simulator import agent_helper_functions
from monopoly_simulator import action_choices
import copy
import random
from collections import Counter

def make_pre_roll_move(player, current_gameboard, allowable_moves, code):
    """
    Decide the best pre-roll action using a simple strategy.
    Skips turn if no specific actions are identified.
    """
    if "skip_turn" in allowable_moves:
        return "skip_turn", {}
    return "concluded_actions", {}


def make_post_roll_move(player, current_gameboard, allowable_moves, code):
    """
    Decide the best post-roll action, including trading logic.
    """
    # Evaluate potential trades if trading is an allowable move
    if "propose_trade" in allowable_moves:
        # Example trade proposal for demonstration purposes
        proposed_trade = {
            "offered_properties": ["Baltic Avenue"],
            "wanted_properties": ["Boardwalk"],
            "cash_offered": 100,
            "cash_requested": 0
        }

        # Evaluate the trade
        if make_trade_decision(player, current_gameboard, proposed_trade):
            return "propose_trade", proposed_trade

    # Default to concluding actions if no other moves are applicable
    if "concluded_actions" in allowable_moves:
        return "concluded_actions", {}

    return "skip_turn", {}


def make_out_of_turn_move(player, current_gameboard, allowable_moves, code):
    """
    Decide the best out-of-turn action.
    """
    if "skip_turn" in allowable_moves:
        return "skip_turn", {}
    return "concluded_actions", {}


def handle_negative_cash_balance(player, current_gameboard):
    """
    Handle negative cash balance using property mortgaging.
    """
    for asset in player.assets:
        if not asset.is_mortgaged:
            return "mortgage_property", {
                "player": player.player_name,
                "asset": asset.name,
                "current_gameboard": "current_gameboard"
            }
    return "declare_bankruptcy", {}


def make_buy_property_decision(player, current_gameboard, asset):
    """
    Decide whether to buy a property based on landing probabilities and financial status.
    """
    # Estimate landing probabilities
    landing_probs = _calculate_landing_probabilities(player, current_gameboard, num_simulations=1000)

    # Evaluate the property based on its landing probability
    property_value = landing_probs.get(asset.name, 0) * 100  # Example: weight probability by 100
    return player.current_cash >= asset.price and property_value > asset.price


def make_bid(player, current_gameboard, asset, current_bid):
    """
    Make a bidding decision for auctions.
    """
    max_bid = player.current_cash // 2
    if current_bid < max_bid:
        return current_bid + (max_bid - current_bid) // 2
    return 0


def make_trade_decision(player, current_gameboard, proposed_trade, num_simulations=1000):
    """
    Use MCMC to evaluate a trade based on landing probabilities.
    :param player: The player considering the trade.
    :param current_gameboard: The current state of the game.
    :param proposed_trade: Dictionary containing details of the trade.
    :param num_simulations: Number of simulations for MCMC.
    :return: True (accept trade) or False (reject trade).
    """
    # Simulate landing probabilities for the player and other players
    player_landing_probs = _calculate_landing_probabilities(player, current_gameboard, num_simulations)
    other_landing_probs = {
        other_player.player_name: _calculate_landing_probabilities(other_player, current_gameboard, num_simulations)
        for other_player in current_gameboard['players']
        if other_player.player_name != player.player_name
    }

    # Evaluate the trade based on landing probabilities
    offered_value = _evaluate_properties(proposed_trade['offered_properties'], player_landing_probs)
    wanted_value = _evaluate_properties(proposed_trade['wanted_properties'], other_landing_probs)

    # Adjust value with cash if applicable
    offered_value += proposed_trade.get('cash_offered', 0)
    wanted_value += proposed_trade.get('cash_requested', 0)

    # Decision: accept the trade if the wanted value exceeds the offered value
    return wanted_value > offered_value


def _calculate_landing_probabilities(player, current_gameboard, num_simulations):
    """
    Simulate landing probabilities for a player using MCMC.
    :param player: The player whose landing probabilities are calculated.
    :param current_gameboard: The current state of the game.
    :param num_simulations: Number of simulations.
    :return: Dictionary with square names as keys and landing probabilities as values.
    """
    position_counts = Counter()
    num_squares = len(current_gameboard['location_sequence'])

    for _ in range(num_simulations):
        current_position = player.current_position

        # Simulate a sequence of moves
        for _ in range(10):  # Simulate the next 10 moves
            dice_roll = random.randint(1, 6) + random.randint(1, 6)  # Roll two dice
            current_position = (current_position + dice_roll) % num_squares
            position_counts[current_position] += 1

    # Normalize counts to probabilities
    total_moves = sum(position_counts.values())
    return {current_gameboard['location_sequence'][pos].name: count / total_moves for pos, count in position_counts.items()}


def _evaluate_properties(properties, landing_probs):
    """
    Evaluate the value of a set of properties based on landing probabilities.
    :param properties: List of property names.
    :param landing_probs: Landing probability distribution.
    :return: Total value of the properties.
    """
    value = 0
    for prop in properties:
        value += landing_probs.get(prop, 0) * 100  # Example: weight landing probability by 100 for property value
    return value


def _build_decision_agent_methods_dict():
    """
    Builds the decision agent methods dictionary for integrating with gameplay.
    """
    return {
        "make_pre_roll_move": make_pre_roll_move,
        "make_post_roll_move": make_post_roll_move,
        "make_out_of_turn_move": make_out_of_turn_move,
        "handle_negative_cash_balance": handle_negative_cash_balance,
        "make_buy_property_decision": make_buy_property_decision,
        "make_bid": make_bid,
        "type": "decision_agent_methods"
    }


decision_agent_methods = _build_decision_agent_methods_dict()
