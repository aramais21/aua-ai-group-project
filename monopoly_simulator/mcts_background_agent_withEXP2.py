from monopoly_simulator import agent_helper_functions  # helper functions are internal to the agent and will not be recorded in the function log.
from monopoly_simulator import diagnostics
from monopoly_simulator.flag_config import flag_config_dict
import logging
import math
import random

logger = logging.getLogger('monopoly_simulator.logging_info.background_agent')

UNSUCCESSFUL_LIMIT = 2
GAMMA = 0.1  # Exploration parameter for EXP3. You can tune this value as needed.

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.weights = []  # Weights for children, used by EXP3

    def expand(self, actions):
        """
        Expand the node by creating child nodes for each available action.
        Initialize all child weights to 1.0.
        """
        for action in actions:
            child = Node(state=self.state, parent=self, action=action)
            self.children.append(child)
        self.weights = [1.0 for _ in self.children]

    def is_leaf(self):
        """
        Check if the node is a leaf node (i.e., has no children).
        """
        return len(self.children) == 0

    def select_child_exp3(self):
        """
        Select a child node based on EXP3 probabilities.
        Returns:
            chosen_index (int): Index of the selected child.
            chosen_probability (float): Probability of selecting the chosen child.
        """
        if not self.children:
            return None, None

        K = len(self.children)
        total_weight = sum(self.weights)
        probabilities = []
        for w in self.weights:
            prob = (1 - GAMMA) * (w / total_weight) + (GAMMA / K)
            probabilities.append(prob)

        # Ensure probabilities sum to 1 due to floating point arithmetic
        probabilities = [p / sum(probabilities) for p in probabilities]

        # Randomly select an arm based on these probabilities
        chosen_index = weighted_choice(probabilities)
        chosen_probability = probabilities[chosen_index]
        return chosen_index, chosen_probability

    def update_weight(self, chosen_index, chosen_probability, reward):
        """
        Update the weight of the chosen child using the EXP3 update rule.

        Args:
            chosen_index (int): Index of the chosen child.
            chosen_probability (float): Probability of selecting the chosen child.
            reward (float): Observed reward from the rollout.
        """
        K = len(self.children)
        if chosen_probability <= 0:
            chosen_probability = 1e-12  # Prevent division by zero

        # Update the weight using the EXP3 update rule
        exponent = (GAMMA * reward) / (chosen_probability * K)
        self.weights[chosen_index] *= math.exp(exponent)


def weighted_choice(probabilities):
    """
    Select an index based on the provided list of probabilities.

    Args:
        probabilities (list of float): Probabilities for each index.

    Returns:
        int: Selected index.
    """
    r = random.random()
    cum_sum = 0.0
    for i, p in enumerate(probabilities):
        cum_sum += p
        if r <= cum_sum:
            return i
    return len(probabilities) - 1  # Fallback due to floating point issues


# ADDED FOR TERMINAL CHECK
def is_terminal_state(current_gameboard):
    """
    Check if the current state of the game is terminal.
    This can be determined by conditions such as:
    - The game_end flag in current_gameboard
    - The number of active players
    - A function from diagnostics or game utilities that determines game completion

    Adjust this logic according to how terminal states are represented in your environment.
    """
    # Example placeholder checks:
    # If there's a 'game_ended' key in current_gameboard:
    if 'game_ended' in current_gameboard and current_gameboard['game_ended']:
        return True

    # Or if diagnostics provides a function to check game completion:
    # if diagnostics.game_completed(current_gameboard):
    #     return True

    # Or check if only one player remains not 'lost':
    active_players = [p for p in current_gameboard['players'] if p.status != 'lost']
    if len(active_players) <= 1:
        return True

    return False


def run_mcts(player, current_gameboard, allowable_moves, max_iterations=100):
    """
    Run MCTS with EXP3-based action selection and multiple depth exploration,
    ensuring each first-level child is visited at least once before deeper expansions.

    Args:
        player: The current player object.
        current_gameboard: The current state of the gameboard.
        allowable_moves (set or list): Set or list of allowable actions.
        max_iterations (int): Maximum number of iterations for MCTS.

    Returns:
        action: The selected action after MCTS.
    """
    # If no moves or single move, no MCTS needed
    if not allowable_moves:
        return None
    if len(allowable_moves) == 1:
        return next(iter(allowable_moves))

    # Check for terminal state early
    # ADDED FOR TERMINAL CHECK
    if is_terminal_state(current_gameboard):
        # No further action needed, game is over; choose a terminal action if any, else None
        return "concluded_actions" if "concluded_actions" in allowable_moves else None

    # Represent state as a simple tuple (player_name, player_cash)
    state = (player.player_name, player.current_cash)

    root = Node(state=state)

    # Expand root with children for each allowable move
    root.expand(list(allowable_moves))

    num_children = len(root.children)
    if max_iterations < num_children:
        logger.warning("max_iterations is less than the number of allowable moves. Increasing max_iterations.")
        max_iterations = num_children + 1  # Ensure at least one iteration after visiting each child

    # Phase 1: Ensure each first-level child gets visited at least once
    for idx, child in enumerate(root.children):
        # Check terminal state before rollout
        # ADDED FOR TERMINAL CHECK
        if is_terminal_state(current_gameboard):
            break

        action = child.action
        reward = rollout_simulation(player, current_gameboard, action)

        # Update child statistics
        child.visits += 1
        child.value += reward

        # Update root's weight for this child using EXP3
        K = num_children
        chosen_probability = 1.0 / K  # Since all children are equally likely initially
        root.update_weight(idx, chosen_probability, reward)

    # Remaining iterations after Phase 1
    remaining_iterations = max_iterations - num_children

    for _ in range(remaining_iterations):
        # Check terminal state before selection and rollout
        # ADDED FOR TERMINAL CHECK
        if is_terminal_state(current_gameboard):
            break

        # SELECTION using EXP3
        node = root
        chosen_indices = []  # To keep track of chosen indices for backpropagation
        chosen_probabilities = []  # To keep track of probabilities

        # Traverse the tree until a leaf node is reached
        while not node.is_leaf():
            chosen_index, chosen_probability = node.select_child_exp3()
            if chosen_index is None:
                break  # No children to select
            chosen_indices.append(chosen_index)
            chosen_probabilities.append(chosen_probability)
            node = node.children[chosen_index]

        # At leaf: ROLLOUT simulation
        # Again check terminal state before rollout
        # ADDED FOR TERMINAL CHECK
        if is_terminal_state(current_gameboard):
            reward = 0.0  # If terminal, no further reward
        else:
            action = node.action
            reward = rollout_simulation(player, current_gameboard, action)

        # BACKPROPAGATION
        node_to_update = root
        for chosen_index, chosen_probability in zip(chosen_indices, chosen_probabilities):
            node_to_update.update_weight(chosen_index, chosen_probability, reward)
            node_to_update = node_to_update.children[chosen_index]

        # Additionally, update the leaf node's statistics
        node.visits += 1
        node.value += reward

    # Choose best action based on the highest average value
    best_child = max(
        root.children,
        key=lambda c: (c.value / c.visits) if c.visits > 0 else 0
    )
    return best_child.action


def rollout_simulation(player, current_gameboard, action):
    """
    Simulate a rollout (playout) for the given action and return a reward.

    Args:
        player: The current player object.
        current_gameboard: The current state of the gameboard.
        action (str): The action to simulate.

    Returns:
        float: The simulated reward for the action.
    """
    # ADDED FOR TERMINAL CHECK
    # If it's a terminal state, no need to proceed with any simulation:
    if is_terminal_state(current_gameboard):
        return 0.0

    # Simple heuristic reward
    if action in ["buy_property", "improve_property", "use_get_out_of_jail_card", "pay_jail_fine"]:
        return random.uniform(0.5, 1.0)
    elif action in ["skip_turn", "concluded_actions"]:
        return random.uniform(-0.1, 0.1)
    elif action in ["mortgage_property", "sell_property", "free_mortgage", "accept_trade_offer",
                    "accept_sell_property_offer", "make_trade_offer", "sell_house_hotel"]:
        return random.uniform(0.0, 0.5)
    else:
        # Default random reward
        return random.uniform(-0.1, 0.1)


def make_pre_roll_move(player, current_gameboard, allowable_moves, code):
    # ADDED FOR TERMINAL CHECK
    if is_terminal_state(current_gameboard):
        logger.debug(f"{player.player_name}: Terminal state detected. No pre-roll move needed.")
        return ("concluded_actions", {}) if "concluded_actions" in allowable_moves else (None, flag_config_dict['successful_action'])

    # Phase game and unsuccessful tries initialization logic
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
        # Run MCTS to decide between skip_turn or concluded_actions if both available
        chosen_action = run_mcts(player, current_gameboard, allowable_moves)
        if chosen_action is not None:
            if chosen_action == "skip_turn":
                player.agent._agent_memory['previous_action'] = "skip_turn"
                return ("skip_turn", {})
            elif chosen_action == "concluded_actions":
                return ("concluded_actions", {})

        if "skip_turn" in allowable_moves:
            player.agent._agent_memory['previous_action'] = "skip_turn"
            return ("skip_turn", {})
        elif "concluded_actions" in allowable_moves:
            return ("concluded_actions", {})
        else:
            logger.error("No valid action found in preroll unsuccessful tries.")
            raise Exception("No valid action found.")

    # If we have multiple possible moves (like getting out of jail), run MCTS:
    chosen_action = run_mcts(player, current_gameboard, allowable_moves)
    if chosen_action is not None:
        param = {
            'player': player.player_name,
            'current_gameboard': "current_gameboard"
        }
        if chosen_action == "use_get_out_of_jail_card":
            player.agent._agent_memory['previous_action'] = "use_get_out_of_jail_card"
            return ("use_get_out_of_jail_card", param)
        elif chosen_action == "pay_jail_fine":
            player.agent._agent_memory['previous_action'] = "pay_jail_fine"
            return ("pay_jail_fine", param)
        elif chosen_action == "skip_turn":
            player.agent._agent_memory['previous_action'] = "skip_turn"
            return ("skip_turn", {})
        elif chosen_action == "concluded_actions":
            return ("concluded_actions", {})

    # Fallback to original logic if MCTS did not choose a specific action
    if player.current_cash >= current_gameboard['go_increment']:
        param = {
            'player': player.player_name,
            'current_gameboard': "current_gameboard"
        }
        if "use_get_out_of_jail_card" in allowable_moves:
            logger.debug(f"{player.player_name}: I am using get out of jail card.")
            player.agent._agent_memory['previous_action'] = "use_get_out_of_jail_card"
            return ("use_get_out_of_jail_card", param)
        elif "pay_jail_fine" in allowable_moves:
            logger.debug(f"{player.player_name}: I am going to pay jail fine.")
            player.agent._agent_memory['previous_action'] = "pay_jail_fine"
            return ("pay_jail_fine", param)

    if "skip_turn" in allowable_moves:
        logger.debug(f"{player.player_name}: I am skipping turn")
        player.agent._agent_memory['previous_action'] = "skip_turn"
        return ("skip_turn", {})
    elif "concluded_actions" in allowable_moves:
        logger.debug(f"{player.player_name}: I am concluding actions")
        return ("concluded_actions", {})
    else:
        logger.error("No valid action found in preroll fallback.")
        raise Exception("No valid action found.")


def make_out_of_turn_move(player, current_gameboard, allowable_moves, code):
    # ADDED FOR TERMINAL CHECK
    if is_terminal_state(current_gameboard):
        logger.debug(f"{player.player_name}: Terminal state detected. No out-of-turn move needed.")
        return ("concluded_actions", {}) if "concluded_actions" in allowable_moves else (None, flag_config_dict['successful_action'])

    # Phase game and unsuccessful tries initialization logic
    for p in current_gameboard['players']:
        if 'phase_game' not in p.agent._agent_memory:
            p.agent._agent_memory['phase_game'] = 1
            p.agent._agent_memory['count_unsuccessful_tries'] = 0

    if player.agent._agent_memory['phase_game'] != 1:
        player.agent._agent_memory['phase_game'] = 1
        player.agent._agent_memory['count_unsuccessful_tries'] = 0

    # Handle failure codes
    if isinstance(code, list):
        code_flag = any(c == flag_config_dict['failure_code'] for c in code)
    else:
        code_flag = (code == flag_config_dict['failure_code'])

    if code_flag:
        player.agent._agent_memory['count_unsuccessful_tries'] += 1
        logger.debug(
            f"{player.player_name} has executed an unsuccessful out of turn action, incrementing unsuccessful_tries counter to {player.agent._agent_memory['count_unsuccessful_tries']}"
        )

    if player.agent._agent_memory['count_unsuccessful_tries'] >= UNSUCCESSFUL_LIMIT:
        logger.debug(f"{player.player_name} has reached out of turn unsuccessful action limits.")
        # Run MCTS to decide between skip_turn or concluded_actions if both available
        chosen_action = run_mcts(player, current_gameboard, allowable_moves)
        if chosen_action is not None:
            if chosen_action == "skip_turn":
                player.agent._agent_memory['previous_action'] = "skip_turn"
                return ("skip_turn", {})
            elif chosen_action == "concluded_actions":
                return ("concluded_actions", {})

        if "skip_turn" in allowable_moves:
            player.agent._agent_memory['previous_action'] = "skip_turn"
            return ("skip_turn", {})
        elif "concluded_actions" in allowable_moves:
            return ("concluded_actions", {})
        else:
            logger.error("No valid action found in out of turn unsuccessful tries.")
            raise Exception("No valid action found.")

    # Run MCTS to pick the best action if multiple moves are available
    chosen_action = run_mcts(player, current_gameboard, allowable_moves)
    if chosen_action is not None:
        param = {
            'player': player.player_name,
            'current_gameboard': "current_gameboard"
        }
        if chosen_action == "accept_trade_offer" and "accept_trade_offer" in allowable_moves:
            player.agent._agent_memory['previous_action'] = "accept_trade_offer"
            return ("accept_trade_offer", param)
        elif chosen_action == "accept_sell_property_offer" and "accept_sell_property_offer" in allowable_moves:
            player.agent._agent_memory['previous_action'] = "accept_sell_property_offer"
            return ("accept_sell_property_offer", param)
        elif chosen_action == "improve_property" and "improve_property" in allowable_moves:
            # We need a param for improve_property from heuristics
            param_improve = agent_helper_functions.identify_improvement_opportunity(player, current_gameboard)
            if param_improve:
                param_improve['player'] = param_improve['player'].player_name
                param_improve['asset'] = param_improve['asset'].name
                param_improve['current_gameboard'] = "current_gameboard"
                player.agent._agent_memory['previous_action'] = "improve_property"
                return ("improve_property", param_improve)

        # Handle free_mortgage if chosen
        if chosen_action == "free_mortgage" and "free_mortgage" in allowable_moves:
            if player.mortgaged_assets:
                player_mortgaged_assets_list = _set_to_sorted_list_mortgaged_assets(player.mortgaged_assets)
                for m in player_mortgaged_assets_list:
                    if player.current_cash - (m.mortgage * (1 + current_gameboard['bank'].mortgage_percentage)) >= current_gameboard['go_increment']:
                        param_free = {
                            'player': player.player_name,
                            'asset': m.name,
                            'current_gameboard': "current_gameboard"
                        }
                        player.agent._agent_memory['previous_action'] = "free_mortgage"
                        return ("free_mortgage", param_free)

    # Fallback to heuristic logic if MCTS did not choose a specific action
    if "accept_trade_offer" in allowable_moves:
        param = {
            'player': player.player_name,
            'current_gameboard': "current_gameboard"
        }
        # Heuristic accept logic as before
        logger.debug(
            f"{player.player_name}: Should I accept the trade offer by {player.outstanding_trade_offer['from_player'].player_name}?"
        )
        logger.debug(
            f"({player.player_name} currently has cash balance of {player.current_cash})"
        )

        if (player.outstanding_trade_offer['cash_offered'] <= 0 and
            len(player.outstanding_trade_offer['property_set_offered']) == 0) and \
           (player.outstanding_trade_offer['cash_wanted'] > 0 or
            len(player.outstanding_trade_offer['property_set_wanted']) > 0):
            pass
        elif (player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer['cash_offered'] > player.current_cash):
            pass
        else:
            reject_flag = 0
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
                    if (player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer['cash_offered']) >= player.current_cash:
                        reject_flag = 1
                    elif (player.current_cash - (player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer['cash_offered']) < current_gameboard['go_increment']/2):
                        reject_flag = 1
                    elif ((player.current_cash - (player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer['cash_offered']) < current_gameboard['go_increment']) and
                          net_offer_worth <= 0):
                        reject_flag = 1
                    else:
                        reject_flag = 0
                elif count_create_new_monopoly - count_lose_existing_monopoly > 0:
                    if (player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer['cash_offered']) >= player.current_cash:
                        reject_flag = 1
                    else:
                        reject_flag = 0

            if reject_flag == 0:
                logger.debug(
                    f"{player.player_name} accepted trade offer from {player.outstanding_trade_offer['from_player'].player_name}"
                )
                logger.debug(
                    f"{player.player_name} received amount = {player.outstanding_trade_offer['cash_offered']} and offered amount = " +
                    f"{player.outstanding_trade_offer['cash_wanted']} during trade"
                )
                player.agent._agent_memory['previous_action'] = "accept_trade_offer"
                return ("accept_trade_offer", param)

    if "accept_sell_property_offer" in allowable_moves:
        param = {
            'player': player.player_name,
            'current_gameboard': "current_gameboard"
        }
        # Original accept_sell_property_offer logic:
        logger.debug(
            f"{player.player_name}: Should I accept the offer by {player.outstanding_property_offer['from_player'].player_name} to buy " +
            f"{player.outstanding_property_offer['asset'].name} for {player.outstanding_property_offer['price']}?"
        )
        logger.debug(
            f"({player.player_name} currently has cash balance of {player.current_cash})"
        )
        if player.outstanding_property_offer['asset'].is_mortgaged or \
           player.outstanding_property_offer['price'] > player.current_cash:
            pass
        elif (player.current_cash - player.outstanding_property_offer['price'] >= current_gameboard['go_increment'] and
              player.outstanding_property_offer['price'] <= player.outstanding_property_offer['asset'].price):
            player.agent._agent_memory['previous_action'] = "accept_sell_property_offer"
            return ("accept_sell_property_offer", param)
        elif agent_helper_functions.will_property_complete_set(player, player.outstanding_property_offer['asset'], current_gameboard):
            if (player.current_cash - player.outstanding_property_offer['price'] >= current_gameboard['go_increment']/2):
                player.agent._agent_memory['previous_action'] = "accept_sell_property_offer"
                return ("accept_sell_property_offer", param)

    if player.status != 'current_move':
        if "improve_property" in allowable_moves:
            param = agent_helper_functions.identify_improvement_opportunity(player, current_gameboard)
            if param:
                if player.agent._agent_memory['previous_action'] == "improve_property" and code == flag_config_dict['failure_code']:
                    pass
                else:
                    logger.debug(
                        f"{player.player_name}: I am going to improve property {param['asset'].name}"
                    )
                    player.agent._agent_memory['previous_action'] = "improve_property"
                    param['player'] = param['player'].player_name
                    param['asset'] = param['asset'].name
                    param['current_gameboard'] = "current_gameboard"
                    return ("improve_property", param)

        if "free_mortgage" in allowable_moves:
            player_mortgaged_assets_list = _set_to_sorted_list_mortgaged_assets(player.mortgaged_assets)
            for m in player_mortgaged_assets_list:
                if player.current_cash - (m.mortgage * (1 + current_gameboard['bank'].mortgage_percentage)) >= current_gameboard['go_increment']:
                    param_free = {
                        'player': player.player_name,
                        'asset': m.name,
                        'current_gameboard': "current_gameboard"
                    }
                    logger.debug(f"{player.player_name}: I am going to free mortgage on {m.name}")
                    player.agent._agent_memory['previous_action'] = "free_mortgage"
                    return ("free_mortgage", param_free)

    else:
        if player.current_cash < current_gameboard['go_increment'] and "make_trade_offer" in allowable_moves:
            potential_offer_list = agent_helper_functions.identify_property_trade_offer_to_player(player, current_gameboard)
            potential_request_list = agent_helper_functions.identify_property_trade_wanted_from_player(player, current_gameboard)
            param_list = agent_helper_functions.curate_trade_offer_multiple_players(
                player, potential_offer_list, potential_request_list, current_gameboard, purpose_flag=1
            )
            if param_list and player.agent._agent_memory['previous_action'] != "make_trade_offer":
                return_action_list = []
                return_param_list = []
                if len(param_list) > 1:
                    logger.debug(f"{player.player_name}: I am going to make multiple trade offers.")
                for param in param_list:
                    logger.debug(
                        f"{player.player_name}: I am making an offer to trade {list(param['offer']['property_set_offered'])[0].name} to " +
                        f"{param['to_player'].player_name} for {param['offer']['cash_wanted']} dollars"
                    )
                    param['from_player'] = param['from_player'].player_name
                    param['to_player'] = param['to_player'].player_name
                    prop_set_offered = {item.name for item in param['offer']['property_set_offered']}
                    prop_set_wanted = {item.name for item in param['offer']['property_set_wanted']}
                    param['offer']['property_set_offered'] = prop_set_offered
                    param['offer']['property_set_wanted'] = prop_set_wanted

                    player.agent._agent_memory['previous_action'] = "make_trade_offer"
                    return_action_list.append("make_trade_offer")
                    return_param_list.append(param)
                return (return_action_list, return_param_list)

        elif "make_trade_offer" in allowable_moves:
            potential_offer_list = agent_helper_functions.identify_property_trade_offer_to_player(player, current_gameboard)
            potential_request_list = agent_helper_functions.identify_property_trade_wanted_from_player(player, current_gameboard)
            param_list = agent_helper_functions.curate_trade_offer_multiple_players(
                player, potential_offer_list, potential_request_list, current_gameboard, purpose_flag=2
            )
            if param_list and player.agent._agent_memory['previous_action'] != "make_trade_offer":
                return_action_list = []
                return_param_list = []
                if len(param_list) > 1:
                    logger.debug(f"{player.player_name}: I am going to make multiple trade offers.")
                for param in param_list:
                    logger.debug(
                        f"{player.player_name}: I am making a trade offer with {param['to_player'].player_name}"
                    )
                    param['from_player'] = param['from_player'].player_name
                    param['to_player'] = param['to_player'].player_name
                    prop_set_offered = {item.name for item in param['offer']['property_set_offered']}
                    prop_set_wanted = {item.name for item in param['offer']['property_set_wanted']}
                    param['offer']['property_set_offered'] = prop_set_offered
                    param['offer']['property_set_wanted'] = prop_set_wanted
                    player.agent._agent_memory['previous_action'] = "make_trade_offer"
                    return_action_list.append("make_trade_offer")
                    return_param_list.append(param)
                return (return_action_list, return_param_list)

    if "skip_turn" in allowable_moves:
        logger.debug(f"{player.player_name}: I am skipping turn")
        player.agent._agent_memory['previous_action'] = "skip_turn"
        return ("skip_turn", {})
    elif "concluded_actions" in allowable_moves:
        logger.debug(f"{player.player_name}: I am concluding actions")
        return ("concluded_actions", {})
    else:
        logger.error("No valid action found in out of turn fallback.")
        raise Exception("No valid action found.")


def make_post_roll_move(player, current_gameboard, allowable_moves, code):
    # ADDED FOR TERMINAL CHECK
    if is_terminal_state(current_gameboard):
        logger.debug(f"{player.player_name}: Terminal state detected. No post-roll move needed.")
        return ("concluded_actions", {}) if "concluded_actions" in allowable_moves else (None, flag_config_dict['successful_action'])

    # Phase game and unsuccessful tries initialization logic
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
        # Run MCTS to decide on concluded_actions
        chosen_action = run_mcts(player, current_gameboard, allowable_moves)
        if chosen_action is not None:
            if chosen_action == "concluded_actions" and "concluded_actions" in allowable_moves:
                return ("concluded_actions", {})
        if "concluded_actions" in allowable_moves:
            return ("concluded_actions", {})
        else:
            logger.error("No valid action found in postroll unsuccessful tries.")
            raise Exception("No valid action found.")

    current_location = current_gameboard['location_sequence'][player.current_position]

    # Use MCTS if multiple moves exist
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
                # Try mortgage or sell to afford property
                to_mortgage = agent_helper_functions.identify_potential_mortgage(player, current_location.price, True)
                if to_mortgage:
                    params['asset'] = to_mortgage.name
                    player.agent._agent_memory['previous_action'] = "mortgage_property"
                    return ("mortgage_property", params)
                else:
                    to_sell = agent_helper_functions.identify_potential_sale(player, current_gameboard, current_location.price, True)
                    if to_sell:
                        params['asset'] = to_sell.name
                        player.agent._agent_memory['previous_action'] = "sell_property"
                        return ("sell_property", params)

        if chosen_action == "concluded_actions" and "concluded_actions" in allowable_moves:
            return ("concluded_actions", {})

    # Fallback to original logic if MCTS did not choose a specific action
    if "buy_property" in allowable_moves:
        if code == flag_config_dict['failure_code']:
            return ("concluded_actions", {})

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
                    to_sell = agent_helper_functions.identify_potential_sale(player, current_gameboard, current_location.price, True)
                    if to_sell:
                        params['asset'] = to_sell.name
                        player.agent._agent_memory['previous_action'] = "sell_property"
                        return ("sell_property", params)

    if "concluded_actions" in allowable_moves:
        return ("concluded_actions", {})
    else:
        logger.error("No valid action found in postroll fallback.")
        raise Exception("No valid action found.")


def make_buy_property_decision(player, current_gameboard, asset):
    """
    Decide whether to buy a property based on current cash and potential monopolies.

    Args:
        player: The current player object.
        current_gameboard: The current state of the gameboard.
        asset: The property asset under consideration.

    Returns:
        bool: True if the player decides to buy the property, False otherwise.
    """
    decision = False
    if player.current_cash - asset.price >= current_gameboard['go_increment']:
        decision = True
    elif asset.price <= player.current_cash and \
         agent_helper_functions.will_property_complete_set(player, asset, current_gameboard):
        decision = True

    return decision


def make_bid(player, current_gameboard, asset, current_bid):
    """
    Make a bid during auctions using MCTS with EXP3-based selection.

    Args:
        player: The current player object.
        current_gameboard: The current state of the gameboard.
        asset: The property asset being bid on.
        current_bid (int): The current highest bid.

    Returns:
        int: The new bid amount, or 0 to pass.
    """
    # Discretize possible bid increments
    possible_bids = []
    increment = (asset.price // 5) if asset.price > 0 else 1
    for i in range(1, 6):
        next_bid = current_bid + i * increment
        if next_bid < player.current_cash:
            possible_bids.append(next_bid)
    if not possible_bids:
        # No possible higher bid than current_bid
        return 0

    # Include the option to pass (bid 0)
    bid_actions = set(str(b) for b in possible_bids)
    bid_actions.add("0")  # Option to pass

    # Run MCTS with a reduced number of iterations for efficiency
    chosen_action = run_mcts(player, current_gameboard, bid_actions, max_iterations=100)
    if chosen_action is not None:
        if chosen_action == "0":
            return 0
        else:
            # Safely convert chosen_action to int
            try:
                return int(float(chosen_action))
            except ValueError:
                logger.error(f"Invalid bid action: {chosen_action}")
                return 0

    # Fallback heuristic if MCTS does not decide
    if current_bid < asset.price:
        new_bid = current_bid + (asset.price - current_bid) // 2
        if new_bid < player.current_cash:
            return new_bid
        else:
            return 0
    elif current_bid < player.current_cash and agent_helper_functions.will_property_complete_set(player, asset, current_gameboard):
        return current_bid + (player.current_cash - current_bid) // 4
    else:
        return 0


def handle_negative_cash_balance(player, current_gameboard):
    """
    Handle situations where the player has a negative cash balance by taking actions to rectify it.

    Args:
        player: The current player object.
        current_gameboard: The current state of the gameboard.

    Returns:
        tuple: (action, parameters) or (None, flag_config_dict['successful_action']) if no action is needed.
    """
    if player.current_cash >= 0:
        return (None, flag_config_dict['successful_action'])

    # Attempt to resolve negative cash by mortgaging, selling properties, etc.
    # Phase 1: Mortgaging unimproved properties
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

    if mortgage_potentials and (max_sum + player.current_cash >= 0):
        # Create allowable actions in the format "mortgage:<property_name>"
        allowable = set(f"mortgage:{p[0].name}" for p in mortgage_potentials)
        chosen_action = run_mcts(player, current_gameboard, allowable, max_iterations=100)
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

    # Phase 2: Selling unimproved properties not in monopolies
    sale_potentials = []
    for a in sorted_player_assets_list:
        if a.color in player.full_color_sets_possessed:
            continue
        elif a.is_mortgaged:
            sale_potentials.append((a, (a.price * current_gameboard['bank'].property_sell_percentage) -
                                     ((1 + current_gameboard['bank'].mortgage_percentage) * a.mortgage)))
        elif a.loc_class == 'real_estate' and (a.num_houses > 0 or a.num_hotels > 0):
            continue
        else:
            sale_potentials.append((a, a.price * current_gameboard['bank'].property_sell_percentage))

    if sale_potentials:
        # Create allowable actions in the format "sell:<property_name>"
        allowable = set(f"sell:{sp[0].name}" for sp in sale_potentials)
        chosen_action = run_mcts(player, current_gameboard, allowable, max_iterations=100)
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

    # Phase 3: Selling houses/hotels from monopolies if needed
    count = 0
    while (player.num_total_houses > 0 or player.num_total_hotels > 0) and count < 3:
        count += 1
        sell_improvement_actions = []
        for a in sorted_player_assets_list:
            if a.loc_class == 'real_estate':
                if a.num_hotels > 0:
                    sell_improvement_actions.append(f"sell_hotel:{a.name}")
                elif a.num_houses > 0:
                    sell_improvement_actions.append(f"sell_house:{a.name}")

        if sell_improvement_actions:
            allowable = set(sell_improvement_actions)
            chosen_action = run_mcts(player, current_gameboard, allowable, max_iterations=100)
            if chosen_action is not None:
                if chosen_action.startswith("sell_house:") or chosen_action.startswith("sell_hotel:"):
                    act, prop_name = chosen_action.split(":")
                    params = {
                        'player': player.player_name,
                        'asset': prop_name,
                        'current_gameboard': "current_gameboard",
                        'sell_house': act == "sell_house",
                        'sell_hotel': act == "sell_hotel"
                    }
                    player.agent._agent_memory['previous_action'] = "sell_house_hotel"
                    return ("sell_house_hotel", params)

    # Phase 4: Final option - sell all properties if needed
    final_sale_assets = sorted_player_assets_list
    if final_sale_assets:
        allowable = set(f"sell:{a.name}" for a in final_sale_assets)
        chosen_action = run_mcts(player, current_gameboard, allowable, max_iterations=100)
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

    return (None, flag_config_dict['successful_action'])


def _set_to_sorted_list_mortgaged_assets(player_mortgaged_assets):
    """
    Convert a set of mortgaged assets to a sorted list based on asset names.

    Args:
        player_mortgaged_assets (set): Set of mortgaged asset objects.

    Returns:
        list: Sorted list of mortgaged asset objects.
    """
    return sorted(player_mortgaged_assets, key=lambda x: x.name)


def _set_to_sorted_list_assets(player_assets):
    """
    Convert a set of assets to a sorted list based on asset names.

    Args:
        player_assets (set): Set of asset objects.

    Returns:
        list: Sorted list of asset objects.
    """
    return sorted(player_assets, key=lambda x: x.name)


def _build_decision_agent_methods_dict():
    """
    Build a dictionary of decision-making methods for the agent.

    Returns:
        dict: Dictionary containing decision-making functions.
    """
    return {
        'handle_negative_cash_balance': handle_negative_cash_balance,
        'make_pre_roll_move': make_pre_roll_move,
        'make_out_of_turn_move': make_out_of_turn_move,
        'make_post_roll_move': make_post_roll_move,
        'make_buy_property_decision': make_buy_property_decision,
        'make_bid': make_bid,
        'type': "decision_agent_methods"
    }


decision_agent_methods = _build_decision_agent_methods_dict()
