from monopoly_simulator import agent_helper_functions
from monopoly_simulator import diagnostics
from monopoly_simulator.flag_config import flag_config_dict
import logging
import random
import copy

logger = logging.getLogger('monopoly_simulator.logging_info.mcmc_agent')
UNSUCCESSFUL_LIMIT = 2

def make_pre_roll_move(player, current_gameboard, allowable_moves, code):
    """
    Uses MCMC to decide pre-roll moves.
    """
    return _mcmc_decision(player, current_gameboard, allowable_moves, code, phase=0)

def make_out_of_turn_move(player, current_gameboard, allowable_moves, code):
    """
    Uses MCMC to decide out-of-turn moves.
    """
    return _mcmc_decision(player, current_gameboard, allowable_moves, code, phase=1)

def make_post_roll_move(player, current_gameboard, allowable_moves, code):
    """
    Uses MCMC to decide post-roll moves.
    """
    return _mcmc_decision(player, current_gameboard, allowable_moves, code, phase=2)

def handle_negative_cash_balance(player, current_gameboard):
    """
    MCMC-driven logic to handle negative cash balance.
    """
    if player.current_cash >= 0:
        return (None, flag_config_dict['successful_action'])

    sorted_assets = sorted(player.assets, key=lambda x: x.price, reverse=True)
    for asset in sorted_assets:
        if player.current_cash >= 0:
            break
        if not asset.is_mortgaged:
            params = {'player': player.player_name, 'asset': asset.name, 'current_gameboard': "current_gameboard"}
            logger.debug(f"{player.player_name} attempting to mortgage {asset.name}")
            return ("mortgage_property", params)
    return (None, flag_config_dict['failure_code'])

def make_buy_property_decision(player, current_gameboard, asset):
    """
    MCMC-based decision to buy properties.
    """
    if player.current_cash >= asset.price + current_gameboard['go_increment']:
        return True
    elif agent_helper_functions.will_property_complete_set(player, asset, current_gameboard):
        return True
    return False

def make_bid(player, current_gameboard, asset, current_bid):
    """
    MCMC-based bidding logic.
    """
    max_bid = player.current_cash // 2
    if current_bid < max_bid:
        return current_bid + (max_bid - current_bid) // 2
    return 0

def _mcmc_decision(player, current_gameboard, allowable_moves, code, phase, num_simulations=100):
    """
    Core MCMC decision logic shared by pre-roll, post-roll, and out-of-turn phases.
    """
    best_action = None
    best_reward = float("-inf")

    for action_name in allowable_moves:
        action_func = getattr(agent_helper_functions, action_name, None)
        if not action_func:
            continue

        params = _generate_parameters(player, current_gameboard, action_name)
        reward = _simulate_action(action_func, params, player, current_gameboard, num_simulations)

        if reward > best_reward:
            best_action = action_name
            best_reward = reward

    if not best_action:
        if "skip_turn" in allowable_moves:
            return "skip_turn", {}
        elif "concluded_actions" in allowable_moves:
            return "concluded_actions", {}
        else:
            raise Exception("No valid actions available.")

    return best_action, _generate_parameters(player, current_gameboard, best_action)

def _generate_parameters(player, current_gameboard, action_name):
    """
    Generate parameters for MCMC simulations.
    """
    if action_name == "buy_property":
        asset = current_gameboard['location_sequence'][player.current_position]
        return {"player": player.player_name, "asset": asset.name, "current_gameboard": "current_gameboard"}
    return {}

def _simulate_action(action_func, parameters, player, current_gameboard, num_simulations):
    """
    Simulate action outcomes using MCMC.
    """
    total_reward = 0
    for _ in range(num_simulations):
        simulated_gameboard = copy.deepcopy(current_gameboard)
        simulated_player = [
            p for p in simulated_gameboard['players'] if p.player_name == player.player_name
        ][0]
        try:
            action_func(simulated_player, simulated_gameboard, **parameters)
            total_reward += _evaluate_game_state(simulated_gameboard, simulated_player)
        except Exception:
            continue

    return total_reward / num_simulations if num_simulations > 0 else 0

def _evaluate_game_state(simulated_gameboard, simulated_player):
    """
    Evaluate the reward for a simulated game state.
    """
    return simulated_player.current_cash + sum(
        asset.price for asset in simulated_player.assets if not asset.is_mortgaged
    )

def _build_decision_agent_methods_dict():
    """
    Builds the decision agent methods dictionary.
    """
    return {
        "handle_negative_cash_balance": handle_negative_cash_balance,
        "make_pre_roll_move": make_pre_roll_move,
        "make_out_of_turn_move": make_out_of_turn_move,
        "make_post_roll_move": make_post_roll_move,
        "make_buy_property_decision": make_buy_property_decision,
        "make_bid": make_bid,
        "type": "decision_agent_methods"
    }

decision_agent_methods = _build_decision_agent_methods_dict()
