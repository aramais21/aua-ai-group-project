from monopoly_simulator import \
    agent_helper_functions  # Helper functions are internal to the agent and will not be recorded in the function log.
from monopoly_simulator import diagnostics
from monopoly_simulator.flag_config import flag_config_dict
import logging
import math
import random
import pickle
import os
import atexit

logger = logging.getLogger('monopoly_simulator.logging_info.background_agent')

UNSUCCESSFUL_LIMIT = 2
K_RAVE = 1000  # RAVE parameter controlling Beta

# Path to save global RAVE statistics
GLOBAL_RAVE_PATH = "global_rave_stats.pkl"


class GlobalRAVE:
    """
    A class to store and manage global RAVE statistics across multiple games.
    """

    def __init__(self):
        # action: (total_reward, count)
        self.rave_values = {}
        self.rave_counts = {}

    def update(self, actions, reward):
        """
        Update RAVE statistics with actions taken and the reward received.
        :param actions: List of actions taken during the game.
        :param reward: Reward received at the end of the game.
        """
        for action in actions:
            if action not in self.rave_values:
                self.rave_values[action] = 0.0
                self.rave_counts[action] = 0
            self.rave_values[action] += reward
            self.rave_counts[action] += 1

    def get_rave_value(self, action):
        """
        Get the average RAVE value for a given action.
        :param action: Action to retrieve the RAVE value for.
        :return: Average RAVE value or a neutral guess if no data.
        """
        if action in self.rave_counts and self.rave_counts[action] > 0:
            return self.rave_values[action] / self.rave_counts[action]
        else:
            # No RAVE data, return neutral (0.5 as a neutral guess)
            return 0.5

    def save(self, path=GLOBAL_RAVE_PATH):
        """
        Save the global RAVE statistics to a file.
        """
        with open(path, 'wb') as f:
            pickle.dump({'values': self.rave_values, 'counts': self.rave_counts}, f)

    def load(self, path=GLOBAL_RAVE_PATH):
        """
        Load the global RAVE statistics from a file.
        """
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.rave_values = data.get('values', {})
                self.rave_counts = data.get('counts', {})
        else:
            logger.info("Global RAVE stats file not found. Starting fresh.")


# Initialize global RAVE
global_rave = GlobalRAVE()
global_rave.load()


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.sum_of_squares = 0.0
        # RAVE statistics: For each action, store cumulative value and count (local RAVE)
        self.rave_values = {}
        self.rave_counts = {}

    def expand(self, actions):
        for a in actions:
            child = Node(self.state, parent=self, action=a)
            self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0

    def best_child(self):
        # Choose child based on Q-RAVE combination
        # Beta = sqrt(K_RAVE/(3N+K_RAVE))
        if self.visits == 0:
            return random.choice(self.children)
        beta = math.sqrt(K_RAVE / (3 * self.visits + K_RAVE))
        best = None
        best_value = -float('inf')
        for c in self.children:
            q_val = 0.0 if c.visits == 0 else (c.value / c.visits)
            rave_val = self.get_rave_value_for_child(c.action) + global_rave.get_rave_value(c.action)
            combined = (1 - beta) * q_val + beta * rave_val
            if combined > best_value:
                best_value = combined
                best = c
        return best

    def get_rave_value_for_child(self, action):
        # Return the average rave value for given action from this node (local RAVE)
        if action in self.rave_counts and self.rave_counts[action] > 0:
            return self.rave_values[action] / self.rave_counts[action]
        else:
            # No local RAVE data, return neutral (0.5 as a neutral guess)
            return 0.5


def run_mcts(player, current_gameboard, allowable_moves, max_iterations=50):
    if not allowable_moves or len(allowable_moves) == 1:
        return None

    state = (player.player_name, player.current_cash)
    root = Node(state=state)
    root.expand(list(allowable_moves))

    all_actions_taken = []

    for _ in range(max_iterations):
        # SELECT
        node = root
        path = [node]
        while not node.is_leaf():
            node = node.best_child()
            path.append(node)

        # ROLLOUT
        action = node.action
        reward, actions_taken = rollout_simulation_with_actions(player, current_gameboard, action)
        all_actions_taken.extend(actions_taken)

        # Clamp reward
        reward = max(0.0, min(1.0, reward))

        # BACKPROPAGATE Q-values
        for n in reversed(path):
            n.visits += 1
            n.value += reward
            n.sum_of_squares += reward ** 2

        # BACKPROPAGATE RAVE updates
        # Update both local and global RAVE
        update_rave(path, actions_taken, reward, global_update=True)

    # Choose best action by visit count
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action


def rollout_simulation_with_actions(player, current_gameboard, action):
    """
    Returns (reward, actions_taken)
    actions_taken is a list of actions chosen during the rollout (including the starting one)
    For simplicity, we treat rollout as random actions from the allowable set.
    """
    actions_taken = [action]
    # We'll do a few random steps
    # The final reward is biased by the initial action.
    if action in ["buy_property", "improve_property", "use_get_out_of_jail_card", "pay_jail_fine"]:
        base_reward = random.uniform(0.5, 1.0)
    elif action in ["skip_turn", "concluded_actions"]:
        base_reward = random.uniform(-0.1, 0.1)
    elif action in ["mortgage_property", "sell_property", "free_mortgage", "accept_trade_offer",
                    "accept_sell_property_offer", "make_trade_offer", "sell_house_hotel"]:
        base_reward = random.uniform(0.0, 0.5)
    else:
        base_reward = random.uniform(-0.1, 0.1)

    # Add a few random hypothetical actions
    possible_extras = ["buy_property", "improve_property", "skip_turn", "mortgage_property", "sell_property"]
    for _ in range(3):
        a = random.choice(possible_extras)
        actions_taken.append(a)
        # Extra random reward not strongly dependent on these actions
        base_reward += random.uniform(-0.05, 0.05)

    # Final reward
    return base_reward, actions_taken


def update_rave(path, actions_taken, reward, global_update=False):
    """
    Update RAVE stats for each node in path with the actions that appeared later.
    :param path: List of nodes from root to leaf.
    :param actions_taken: List of actions taken during the rollout.
    :param reward: Reward received.
    :param global_update: If True, update global RAVE stats.
    """
    # Update local RAVE stats for each node in path
    for i, node in enumerate(path):
        # Actions after this node
        for a in actions_taken[i + 1:]:
            if a not in node.rave_values:
                node.rave_values[a] = 0.0
                node.rave_counts[a] = 0
            node.rave_values[a] += reward
            node.rave_counts[a] += 1

    if global_update:
        # Update global RAVE stats once per simulation
        global_rave.update(actions_taken, reward)


def make_pre_roll_move(player, current_gameboard, allowable_moves, code):
    for p in current_gameboard['players']:
        if 'phase_game' not in p.agent._agent_memory:
            p.agent._agent_memory['phase_game'] = 0
            p.agent._agent_memory['count_unsuccessful_tries'] = 0

    if player.agent._agent_memory['phase_game'] != 0:
        player.agent._agent_memory['phase_game'] = 0
        for p in current_gameboard['players']:
            if p.status != 'lost':
                p.agent._agent_memory['count_unsuccessful_tries'] = 0

    if code == flag_config_dict['failure_code']:
        player.agent._agent_memory['count_unsuccessful_tries'] += 1
        logger.debug(
            f"{player.player_name} has executed an unsuccessful preroll action, incrementing unsuccessful_tries counter to {player.agent._agent_memory['count_unsuccessful_tries']}"
        )

    if player.agent._agent_memory['count_unsuccessful_tries'] >= UNSUCCESSFUL_LIMIT:
        logger.debug(f"{player.player_name} has reached preroll unsuccessful action limits.")
        chosen_action = run_mcts(player, current_gameboard, allowable_moves)
        if chosen_action is not None:
            if chosen_action == "skip_turn":
                player.agent._agent_memory['previous_action'] = "skip_turn"
                return ("skip_turn", dict())
            elif chosen_action == "concluded_actions":
                return ("concluded_actions", dict())

        if "skip_turn" in allowable_moves:
            player.agent._agent_memory['previous_action'] = "skip_turn"
            return ("skip_turn", dict())
        elif "concluded_actions" in allowable_moves:
            return ("concluded_actions", dict())
        else:
            logger.error("No valid action available and no fallback option.")
            raise Exception("No valid action available and no fallback option.")

    chosen_action = run_mcts(player, current_gameboard, allowable_moves)
    if chosen_action is not None:
        param = {'player': player.player_name, 'current_gameboard': "current_gameboard"}
        if chosen_action == "use_get_out_of_jail_card":
            player.agent._agent_memory['previous_action'] = "use_get_out_of_jail_card"
            return ("use_get_out_of_jail_card", param)
        elif chosen_action == "pay_jail_fine":
            player.agent._agent_memory['previous_action'] = "pay_jail_fine"
            return ("pay_jail_fine", param)
        elif chosen_action == "skip_turn":
            player.agent._agent_memory['previous_action'] = "skip_turn"
            return ("skip_turn", dict())
        elif chosen_action == "concluded_actions":
            return ("concluded_actions", dict())

    if player.current_cash >= current_gameboard['go_increment']:
        param = {'player': player.player_name, 'current_gameboard': "current_gameboard"}
        if "use_get_out_of_jail_card" in allowable_moves:
            player.agent._agent_memory['previous_action'] = "use_get_out_of_jail_card"
            return ("use_get_out_of_jail_card", param)
        elif "pay_jail_fine" in allowable_moves:
            player.agent._agent_memory['previous_action'] = "pay_jail_fine"
            return ("pay_jail_fine", param)

    if "skip_turn" in allowable_moves:
        player.agent._agent_memory['previous_action'] = "skip_turn"
        return ("skip_turn", dict())
    elif "concluded_actions" in allowable_moves:
        return ("concluded_actions", dict())
    else:
        logger.error("No valid action available and no fallback option.")
        raise Exception("No valid action available and no fallback option.")


def make_out_of_turn_move(player, current_gameboard, allowable_moves, code):
    for p in current_gameboard['players']:
        if 'phase_game' not in p.agent._agent_memory:
            p.agent._agent_memory['phase_game'] = 1
            p.agent._agent_memory['count_unsuccessful_tries'] = 0

    if player.agent._agent_memory['phase_game'] != 1:
        player.agent._agent_memory['phase_game'] = 1
        player.agent._agent_memory['count_unsuccessful_tries'] = 0

    if isinstance(code, list):
        code_flag = any(c == flag_config_dict['failure_code'] for c in code)
        if code_flag:
            player.agent._agent_memory['count_unsuccessful_tries'] += 1
            logger.debug(
                f"{player.player_name} has executed an unsuccessful out of turn action, incrementing unsuccessful_tries counter to {player.agent._agent_memory['count_unsuccessful_tries']}"
            )
    elif code == flag_config_dict['failure_code']:
        player.agent._agent_memory['count_unsuccessful_tries'] += 1
        logger.debug(
            f"{player.player_name} has executed an unsuccessful out of turn action, incrementing unsuccessful_tries counter to {player.agent._agent_memory['count_unsuccessful_tries']}"
        )

    if player.agent._agent_memory['count_unsuccessful_tries'] >= UNSUCCESSFUL_LIMIT:
        logger.debug(f"{player.player_name} has reached out of turn unsuccessful action limits.")
        chosen_action = run_mcts(player, current_gameboard, allowable_moves)
        if chosen_action is not None:
            if chosen_action == "skip_turn":
                player.agent._agent_memory['previous_action'] = "skip_turn"
                return ("skip_turn", dict())
            elif chosen_action == "concluded_actions":
                return ("concluded_actions", dict())

        if "skip_turn" in allowable_moves:
            player.agent._agent_memory['previous_action'] = "skip_turn"
            return ("skip_turn", dict())
        elif "concluded_actions" in allowable_moves:
            return ("concluded_actions", dict())
        else:
            logger.error("No valid action available and no fallback option.")
            raise Exception("No valid action available and no fallback option.")

    chosen_action = run_mcts(player, current_gameboard, allowable_moves)
    if chosen_action is not None:
        param = {'player': player.player_name, 'current_gameboard': "current_gameboard"}
        if chosen_action == "accept_trade_offer" and "accept_trade_offer" in allowable_moves:
            player.agent._agent_memory['previous_action'] = "accept_trade_offer"
            return ("accept_trade_offer", param)
        elif chosen_action == "accept_sell_property_offer" and "accept_sell_property_offer" in allowable_moves:
            player.agent._agent_memory['previous_action'] = "accept_sell_property_offer"
            return ("accept_sell_property_offer", param)
        elif chosen_action == "improve_property" and "improve_property" in allowable_moves:
            param_improve = agent_helper_functions.identify_improvement_opportunity(player, current_gameboard)
            if param_improve:
                param_improve['player'] = param_improve['player'].player_name
                param_improve['asset'] = param_improve['asset'].name
                param_improve['current_gameboard'] = "current_gameboard"
                player.agent._agent_memory['previous_action'] = "improve_property"
                return ("improve_property", param_improve)
        elif chosen_action == "free_mortgage" and "free_mortgage" in allowable_moves:
            if player.mortgaged_assets:
                player_mortgaged_assets_list = _set_to_sorted_list_mortgaged_assets(player.mortgaged_assets)
                for m in player_mortgaged_assets_list:
                    if player.current_cash - (m.mortgage * (1 + current_gameboard['bank'].mortgage_percentage)) >= \
                            current_gameboard['go_increment']:
                        param_free = {
                            'player': player.player_name,
                            'asset': m.name,
                            'current_gameboard': "current_gameboard"
                        }
                        player.agent._agent_memory['previous_action'] = "free_mortgage"
                        return ("free_mortgage", param_free)

    # Fallback logic from original code
    if "accept_trade_offer" in allowable_moves:
        param = {'player': player.player_name, 'current_gameboard': "current_gameboard"}
        # Original heuristic accept logic
        logger.debug(
            f"{player.player_name}: Should I accept the trade offer by {player.outstanding_trade_offer['from_player'].player_name}?"
        )
        logger.debug(f"({player.player_name} currently has cash balance of {player.current_cash})")

        reject_flag = 0
        if (player.outstanding_trade_offer['cash_offered'] <= 0 and len(
                player.outstanding_trade_offer['property_set_offered']) == 0) and \
                (player.outstanding_trade_offer['cash_wanted'] > 0 or len(
                    player.outstanding_trade_offer['property_set_wanted']) > 0):
            pass
        elif player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer[
            'cash_offered'] > player.current_cash:
            pass
        else:
            offered_properties_net_worth = 0
            wanted_properties_net_worth = 0
            for prop in player.outstanding_trade_offer['property_set_wanted']:
                if prop.is_mortgaged:
                    reject_flag = 1
                    break
                else:
                    wanted_properties_net_worth += prop.price

            if reject_flag == 0:
                for prop in player.outstanding_trade_offer['property_set_offered']:
                    if prop.is_mortgaged:
                        reject_flag = 1
                        break
                    else:
                        offered_properties_net_worth += prop.price

            if reject_flag == 0:
                net_offer_worth = (offered_properties_net_worth + player.outstanding_trade_offer['cash_offered']) - \
                                  (wanted_properties_net_worth + player.outstanding_trade_offer['cash_wanted'])
                net_amount_requested = -1 * net_offer_worth
                count_create_new_monopoly = 0
                count_lose_existing_monopoly = 0
                for prop in player.outstanding_trade_offer['property_set_offered']:
                    if agent_helper_functions.will_property_complete_set(player, prop, current_gameboard):
                        count_create_new_monopoly += 1
                for prop in player.outstanding_trade_offer['property_set_wanted']:
                    if prop.color in player.full_color_sets_possessed:
                        count_lose_existing_monopoly += 1

                if count_lose_existing_monopoly - count_create_new_monopoly > 0:
                    reject_flag = 1
                elif count_lose_existing_monopoly - count_create_new_monopoly == 0:
                    if (player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer[
                        'cash_offered']) >= player.current_cash:
                        reject_flag = 1
                    elif player.current_cash - (
                            player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer[
                        'cash_offered']) < current_gameboard['go_increment'] / 2:
                        reject_flag = 1
                    elif (player.current_cash - (
                            player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer[
                        'cash_offered']) < current_gameboard['go_increment']) \
                            and net_offer_worth <= 0:
                        reject_flag = 1
                    else:
                        reject_flag = 0
                elif count_create_new_monopoly - count_lose_existing_monopoly > 0:
                    if (player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer[
                        'cash_offered']) >= player.current_cash:
                        reject_flag = 1
                    else:
                        reject_flag = 0

        if reject_flag == 0:
            player.agent._agent_memory['previous_action'] = "accept_trade_offer"
            return ("accept_trade_offer", param)

    if "accept_sell_property_offer" in allowable_moves:
        param = {'player': player.player_name, 'current_gameboard': "current_gameboard"}
        offer = player.outstanding_property_offer['asset']
        if offer.is_mortgaged or player.outstanding_property_offer['price'] > player.current_cash:
            pass
        elif player.current_cash - player.outstanding_property_offer['price'] >= current_gameboard['go_increment'] and \
                player.outstanding_property_offer['price'] <= player.outstanding_property_offer['asset'].price:
            player.agent._agent_memory['previous_action'] = "accept_sell_property_offer"
            return ("accept_sell_property_offer", param)
        elif agent_helper_functions.will_property_complete_set(player, player.outstanding_property_offer['asset'],
                                                               current_gameboard):
            if player.current_cash - player.outstanding_property_offer['price'] >= current_gameboard[
                'go_increment'] / 2:
                player.agent._agent_memory['previous_action'] = "accept_sell_property_offer"
                return ("accept_sell_property_offer", param)

    if player.status != 'current_move':
        if "improve_property" in allowable_moves:
            param = agent_helper_functions.identify_improvement_opportunity(player, current_gameboard)
            if param:
                if player.agent._agent_memory['previous_action'] == "improve_property" and code == flag_config_dict[
                    'failure_code']:
                    pass
                else:
                    player.agent._agent_memory['previous_action'] = "improve_property"
                    param['player'] = param['player'].player_name
                    param['asset'] = param['asset'].name
                    param['current_gameboard'] = "current_gameboard"
                    return ("improve_property", param)

        if player.mortgaged_assets:
            player_mortgaged_assets_list = _set_to_sorted_list_mortgaged_assets(player.mortgaged_assets)
            for m in player_mortgaged_assets_list:
                if player.current_cash - (m.mortgage * (1 + current_gameboard['bank'].mortgage_percentage)) >= \
                        current_gameboard['go_increment'] and "free_mortgage" in allowable_moves:
                    param = {
                        'player': player.player_name,
                        'asset': m.name,
                        'current_gameboard': "current_gameboard"
                    }
                    player.agent._agent_memory['previous_action'] = "free_mortgage"
                    return ("free_mortgage", param)

    if player.status != 'current_move':
        # Further logic can be implemented here if needed
        pass

    # If no MCTS preference or fallback triggered:
    if "skip_turn" in allowable_moves:
        player.agent._agent_memory['previous_action'] = "skip_turn"
        return ("skip_turn", dict())
    elif "concluded_actions" in allowable_moves:
        return ("concluded_actions", dict())
    else:
        logger.error("No valid action available and no fallback option.")
        raise Exception("No valid action available and no fallback option.")


def make_post_roll_move(player, current_gameboard, allowable_moves, code):
    for p in current_gameboard['players']:
        if 'phase_game' not in p.agent._agent_memory:
            p.agent._agent_memory['phase_game'] = 2
            p.agent._agent_memory['count_unsuccessful_tries'] = 0

    if player.agent._agent_memory['phase_game'] != 2:
        player.agent._agent_memory['phase_game'] = 2
        for p in current_gameboard['players']:
            if p.status != 'lost':
                p.agent._agent_memory['count_unsuccessful_tries'] = 0

    if code == flag_config_dict['failure_code']:
        player.agent._agent_memory['count_unsuccessful_tries'] += 1
        logger.debug(
            f"{player.player_name} has executed an unsuccessful postroll action, incrementing unsuccessful_tries counter to {player.agent._agent_memory['count_unsuccessful_tries']}"
        )

    if player.agent._agent_memory['count_unsuccessful_tries'] >= UNSUCCESSFUL_LIMIT:
        logger.debug(f"{player.player_name} has reached postroll unsuccessful action limits.")
        chosen_action = run_mcts(player, current_gameboard, allowable_moves)
        if chosen_action is not None:
            if chosen_action == "concluded_actions" and "concluded_actions" in allowable_moves:
                return ("concluded_actions", dict())

        if "concluded_actions" in allowable_moves:
            return ("concluded_actions", dict())
        else:
            logger.error("No valid action available and no fallback option.")
            raise Exception("No valid action available and no fallback option.")

    current_location = current_gameboard['location_sequence'][player.current_position]

    chosen_action = run_mcts(player, current_gameboard, allowable_moves)
    if chosen_action is not None:
        if chosen_action == "buy_property" and "buy_property" in allowable_moves:
            params = {
                'player': player.player_name,
                'asset': current_location.name,
                'current_gameboard': "current_gameboard"
            }
            if make_buy_property_decision(player, current_gameboard, current_location):
                player.agent._agent_memory['previous_action'] = "buy_property"
                return ("buy_property", params)
            else:
                # Fallback mortgage/sell logic
                to_mortgage = agent_helper_functions.identify_potential_mortgage(player, current_location.price, True)
                if to_mortgage:
                    params['asset'] = to_mortgage.name
                    player.agent._agent_memory['previous_action'] = "mortgage_property"
                    return ("mortgage_property", params)
                else:
                    to_sell = agent_helper_functions.identify_potential_sale(player, current_gameboard,
                                                                             current_location.price, True)
                    if to_sell:
                        params['asset'] = to_sell.name
                        player.agent._agent_memory['previous_action'] = "sell_property"
                        return ("sell_property", params)

        if chosen_action == "concluded_actions" and "concluded_actions" in allowable_moves:
            return ("concluded_actions", dict())

    # Fallback logic if MCTS not giving final solution
    if "buy_property" in allowable_moves:
        if code == flag_config_dict['failure_code']:
            return ("concluded_actions", dict())

        params = {
            'player': player.player_name,
            'asset': current_location.name,
            'current_gameboard': "current_gameboard"
        }

        if make_buy_property_decision(player, current_gameboard, current_location):
            player.agent._agent_memory['previous_action'] = "buy_property"
            return ("buy_property", params)
        else:
            if agent_helper_functions.will_property_complete_set(player, current_location, current_gameboard):
                to_mortgage = agent_helper_functions.identify_potential_mortgage(player, current_location.price, True)
                if to_mortgage:
                    params['asset'] = to_mortgage.name
                    player.agent._agent_memory['previous_action'] = "mortgage_property"
                    return ("mortgage_property", params)
                else:
                    to_sell = agent_helper_functions.identify_potential_sale(player, current_gameboard,
                                                                             current_location.price, True)
                    if to_sell:
                        params['asset'] = to_sell.name
                        player.agent._agent_memory['previous_action'] = "sell_property"
                        return ("sell_property", params)

    if "concluded_actions" in allowable_moves:
        return ("concluded_actions", dict())
    else:
        logger.error("No valid action available and no fallback option.")
        raise Exception("No valid action available and no fallback option.")


def make_buy_property_decision(player, current_gameboard, asset):
    decision = False
    if player.current_cash - asset.price >= current_gameboard['go_increment']:
        decision = True
    elif asset.price <= player.current_cash and agent_helper_functions.will_property_complete_set(player, asset,
                                                                                                  current_gameboard):
        decision = True
    return decision


def make_bid(player, current_gameboard, asset, current_bid):
    increment = (asset.price // 5) if asset.price > 0 else 1
    possible_bids = []
    for i in range(1, 6):
        next_bid = current_bid + i * increment
        if next_bid < player.current_cash:
            possible_bids.append(next_bid)
    if not possible_bids:
        return 0

    bid_actions = set(str(b) for b in possible_bids)
    bid_actions.add("0")  # Pass
    chosen_action = run_mcts(player, current_gameboard, bid_actions, max_iterations=30)
    if chosen_action is not None:
        if chosen_action == "0":
            return 0
        else:
            try:
                return int(float(chosen_action))
            except ValueError:
                logger.error(f"Invalid bid action: {chosen_action}")
                return 0

    # Fallback heuristic
    if current_bid < asset.price:
        new_bid = current_bid + (asset.price - current_bid) // 2
        if new_bid < player.current_cash:
            return new_bid
        else:
            return 0
    elif current_bid < player.current_cash and agent_helper_functions.will_property_complete_set(player, asset,
                                                                                                 current_gameboard):
        return current_bid + (player.current_cash - current_bid) // 4
    else:
        return 0


def handle_negative_cash_balance(player, current_gameboard):
    if player.current_cash >= 0:
        return (None, flag_config_dict['successful_action'])

    sorted_player_assets_list = _set_to_sorted_list_assets(player.assets)
    mortgage_potentials = []
    max_sum = 0
    for a in sorted_player_assets_list:
        if a.is_mortgaged:
            continue
        elif a.loc_class == 'real_estate' and (a.num_houses > 0 or a.num_hotels > 0):
            continue
        else:
            mortgage_potentials.append((a, a.mortgage))
            max_sum += a.mortgage

    if mortgage_potentials and max_sum + player.current_cash >= 0:
        sorted_potentials = sorted(mortgage_potentials, key=lambda x: x[1])
        if len(sorted_potentials) > 1:
            allowable = set(f"mortgage:{p[0].name}" for p in sorted_potentials)
            chosen_action = run_mcts(player, current_gameboard, allowable, max_iterations=30)
            if chosen_action is not None:
                if chosen_action.startswith("mortgage:"):
                    chosen_property = chosen_action.split(":")[1]
                    params = {
                        'player': player.player_name,
                        'asset': chosen_property,
                        'current_gameboard': "current_gameboard"
                    }
                    player.agent._agent_memory['previous_action'] = "mortgage_property"
                    return ("mortgage_property", params)

        for p in sorted_potentials:
            if player.current_cash >= 0:
                return (None, flag_config_dict['successful_action'])
            params = {
                'player': player.player_name,
                'asset': p[0].name,
                'current_gameboard': "current_gameboard"
            }
            player.agent._agent_memory['previous_action'] = "mortgage_property"
            return ("mortgage_property", params)

    sale_potentials = []
    for a in sorted_player_assets_list:
        if a.color in player.full_color_sets_possessed:
            continue
        elif a.is_mortgaged:
            sale_potentials.append((a, (a.price * current_gameboard['bank'].property_sell_percentage) - (
                        (1 + current_gameboard['bank'].mortgage_percentage) * a.mortgage)))
        elif a.loc_class == 'real_estate' and (a.num_houses > 0 or a.num_hotels > 0):
            continue
        else:
            sale_potentials.append((a, a.price * current_gameboard['bank'].property_sell_percentage))

    if sale_potentials:
        sorted_sp = sorted(sale_potentials, key=lambda x: x[1])
        if len(sorted_sp) > 1:
            allowable = set(f"sell:{p[0].name}" for p in sorted_sp)
            chosen_action = run_mcts(player, current_gameboard, allowable, max_iterations=30)
            if chosen_action is not None:
                if chosen_action.startswith("sell:"):
                    chosen_property = chosen_action.split(":")[1]
                    params = {
                        'player': player.player_name,
                        'asset': chosen_property,
                        'current_gameboard': "current_gameboard"
                    }
                    player.agent._agent_memory['previous_action'] = "sell_property"
                    return ("sell_property", params)

        for p in sorted_sp:
            if player.current_cash >= 0:
                return (None, flag_config_dict['successful_action'])
            params = {
                'player': player.player_name,
                'asset': p[0].name,
                'current_gameboard': "current_gameboard"
            }
            player.agent._agent_memory['previous_action'] = "sell_property"
            return ("sell_property", params)

    count = 0
    while (player.num_total_houses > 0 or player.num_total_hotels > 0) and count < 3:
        count += 1
        sorted_assets_list = _set_to_sorted_list_assets(player.assets)
        sell_improvement_actions = []
        for a in sorted_assets_list:
            if a.loc_class == 'real_estate':
                if a.num_hotels > 0:
                    sell_improvement_actions.append(("sell_hotel:" + a.name, a))
                elif a.num_houses > 0:
                    sell_improvement_actions.append(("sell_house:" + a.name, a))

        if sell_improvement_actions:
            if len(sell_improvement_actions) > 1:
                allowable = set(x[0] for x in sell_improvement_actions)
                chosen_action = run_mcts(player, current_gameboard, allowable, max_iterations=30)
                if chosen_action is not None:
                    act, prop_name = chosen_action.split(":")
                    params = {
                        'player': player.player_name,
                        'asset': prop_name,
                        'current_gameboard': "current_gameboard"
                    }
                    if act == "sell_house":
                        params['sell_house'] = True
                        params['sell_hotel'] = False
                    else:
                        params['sell_house'] = False
                        params['sell_hotel'] = True
                    player.agent._agent_memory['previous_action'] = "sell_house_hotel"
                    return ("sell_house_hotel", params)
            else:
                act, prop_name = sell_improvement_actions[0][0].split(":")
                if player.current_cash >= 0:
                    return (None, flag_config_dict['successful_action'])
                params = {
                    'player': player.player_name,
                    'asset': prop_name,
                    'current_gameboard': "current_gameboard"
                }
                if act == "sell_house":
                    params['sell_house'] = True
                    params['sell_hotel'] = False
                else:
                    params['sell_house'] = False
                    params['sell_hotel'] = True
                player.agent._agent_memory['previous_action'] = "sell_house_hotel"
                return ("sell_house_hotel", params)

    final_sale_assets = player.assets.copy()
    final_assets_list = _set_to_sorted_list_assets(final_sale_assets)
    if final_assets_list:
        if len(final_assets_list) > 1:
            allowable = set(f"sell:{a.name}" for a in final_assets_list)
            chosen_action = run_mcts(player, current_gameboard, allowable, max_iterations=30)
            if chosen_action is not None:
                if chosen_action.startswith("sell:"):
                    chosen_property = chosen_action.split(":")[1]
                    params = {
                        'player': player.player_name,
                        'asset': chosen_property,
                        'current_gameboard': "current_gameboard"
                    }
                    player.agent._agent_memory['previous_action'] = "sell_property"
                    return ("sell_property", params)
        for a in final_assets_list:
            if player.current_cash >= 0:
                return (None, flag_config_dict['successful_action'])
            params = {
                'player': player.player_name,
                'asset': a.name,
                'current_gameboard': "current_gameboard"
            }
            player.agent._agent_memory['previous_action'] = "sell_property"
            return ("sell_property", params)

    return (None, flag_config_dict['successful_action'])


def _set_to_sorted_list_mortgaged_assets(player_mortgaged_assets):
    player_m_assets_list = list(player_mortgaged_assets)
    player_m_assets_list.sort(key=lambda x: x.name)
    return player_m_assets_list


def _set_to_sorted_list_assets(player_assets):
    player_assets_list = list(player_assets)
    player_assets_list.sort(key=lambda x: x.name)
    return player_assets_list


def _build_decision_agent_methods_dict():
    ans = dict()
    ans['handle_negative_cash_balance'] = handle_negative_cash_balance
    ans['make_pre_roll_move'] = make_pre_roll_move
    ans['make_out_of_turn_move'] = make_out_of_turn_move
    ans['make_post_roll_move'] = make_post_roll_move
    ans['make_buy_property_decision'] = make_buy_property_decision
    ans['make_bid'] = make_bid
    ans['type'] = "decision_agent_methods"
    return ans


decision_agent_methods = _build_decision_agent_methods_dict()


# Save global RAVE statistics periodically or at the end of all games
def save_global_rave_stats():
    global_rave.save()


# Ensure that global RAVE stats are saved when the program exits
atexit.register(save_global_rave_stats)
