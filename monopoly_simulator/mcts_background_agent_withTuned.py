from monopoly_simulator import agent_helper_functions # helper functions are internal to the agent and will not be recorded in the function log.
from monopoly_simulator import diagnostics
from monopoly_simulator.flag_config import flag_config_dict
import logging
import math
import random

logger = logging.getLogger('monopoly_simulator.logging_info.background_agent')

UNSUCCESSFUL_LIMIT = 2

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.sum_of_squares = 0.0  # for variance calculation

    def ucb1_tuned_value(self):
        if self.visits == 0:
            return float('inf')
        average = self.value / self.visits
        # Calculate variance term
        variance = (self.sum_of_squares / self.visits) - (average ** 2)
        if variance < 0:
            variance = 0  # floating point protection
        # parent's total visits
        n = self.parent.visits
        N_j = self.visits
        ln_n = math.log(n)
        # UCB1-Tuned formula
        exploration = math.sqrt((ln_n / N_j) * min(0.25, variance + math.sqrt((2 * ln_n) / N_j)))
        return average + exploration

    def best_child(self):
        # choose child with max UCB1-Tuned value
        return max(self.children, key=lambda c: c.ucb1_tuned_value())

    def expand(self, actions):
        # expand by creating child nodes for each action
        for a in actions:
            child = Node(self.state, parent=self, action=a)
            self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0


def run_mcts(player, current_gameboard, allowable_moves, max_iterations=50):
    """
    Run MCTS with UCB1-Tuned selection policy on the current decision point.
    This is a simplified MCTS that uses mock rollouts and random reward heuristics.
    """
    if not allowable_moves or len(allowable_moves) == 1:
        return None

    # Represent state as a simple tuple (player_name, player_cash)
    state = (player.player_name, player.current_cash)
    root = Node(state=state)
    root.expand(list(allowable_moves))

    for _ in range(max_iterations):
        # SELECT
        node = root
        while not node.is_leaf():
            node = node.best_child()

        # SIMULATE (rollout)
        action = node.action
        reward = rollout_simulation(player, current_gameboard, action)
        # Clamp reward to [0,1]
        reward = max(0.0, min(1.0, reward))

        # BACKPROPAGATE
        while node is not None:
            node.visits += 1
            node.value += reward
            node.sum_of_squares += reward ** 2
            node = node.parent

    # Choose best action based on visits
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action


def rollout_simulation(player, current_gameboard, action):
    # Simple heuristic to assign random reward
    if action in ["buy_property", "improve_property", "use_get_out_of_jail_card", "pay_jail_fine"]:
        return random.uniform(0.5, 1.0)
    elif action in ["skip_turn", "concluded_actions"]:
        return random.uniform(-0.1, 0.1)
    elif action in ["mortgage_property", "sell_property", "free_mortgage", "accept_trade_offer",
                    "accept_sell_property_offer", "make_trade_offer", "sell_house_hotel"]:
        return random.uniform(0.0, 0.5)
    else:
        return random.uniform(-0.1, 0.1)


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
        logger.debug(player.player_name + ' has executed an unsuccessful preroll action, incrementing unsuccessful_tries ' +
                                          'counter to ' + str(player.agent._agent_memory['count_unsuccessful_tries']))

    if player.agent._agent_memory['count_unsuccessful_tries'] >= UNSUCCESSFUL_LIMIT:
        logger.debug(player.player_name + ' has reached preroll unsuccessful action limits.')
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
            logger.error("Exception")
            raise Exception

    chosen_action = run_mcts(player, current_gameboard, allowable_moves)
    if chosen_action is not None:
        param = dict()
        param['player'] = player.player_name
        param['current_gameboard'] = "current_gameboard"
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
        param = dict()
        param['player'] = player.player_name
        param['current_gameboard'] = "current_gameboard"
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
        logger.error("Exception")
        raise Exception


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
            logger.debug(player.player_name + ' has executed an unsuccessful out of turn action, ' +
                         f'incrementing unsuccessful_tries counter to {player.agent._agent_memory["count_unsuccessful_tries"]}')
    elif code == flag_config_dict['failure_code']:
        player.agent._agent_memory['count_unsuccessful_tries'] += 1
        logger.debug(player.player_name + ' has executed an unsuccessful out of turn action, ' +
                     f'incrementing unsuccessful_tries counter to {player.agent._agent_memory["count_unsuccessful_tries"]}')

    if player.agent._agent_memory['count_unsuccessful_tries'] >= UNSUCCESSFUL_LIMIT:
        logger.debug(player.player_name + ' has reached out of turn unsuccessful action limits.')
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
            logger.error("Exception")
            raise Exception

    chosen_action = run_mcts(player, current_gameboard, allowable_moves)
    if chosen_action is not None:
        param = dict()
        param['player'] = player.player_name
        param['current_gameboard'] = "current_gameboard"

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

        if chosen_action == "free_mortgage" and "free_mortgage" in allowable_moves:
            if player.mortgaged_assets:
                player_mortgaged_assets_list = _set_to_sorted_list_mortgaged_assets(player.mortgaged_assets)
                for m in player_mortgaged_assets_list:
                    if player.current_cash - (m.mortgage*(1+current_gameboard['bank'].mortgage_percentage)) >= current_gameboard['go_increment']:
                        param_free = dict()
                        param_free['player'] = player.player_name
                        param_free['asset'] = m.name
                        param_free['current_gameboard'] = "current_gameboard"
                        player.agent._agent_memory['previous_action'] = "free_mortgage"
                        return ("free_mortgage", param_free)

    # fallback logic unchanged
    if "accept_trade_offer" in allowable_moves:
        param = dict()
        param['player'] = player.player_name
        param['current_gameboard'] = "current_gameboard"
        # original heuristic accept logic
        logger.debug(player.player_name+ ': Should I accept the trade offer by '+player.outstanding_trade_offer['from_player'].player_name+'?')
        logger.debug('('+player.player_name+' currently has cash balance of '+str(player.current_cash)+')')

        if (player.outstanding_trade_offer['cash_offered'] <= 0 and len(player.outstanding_trade_offer['property_set_offered'])==0) and \
                (player.outstanding_trade_offer['cash_wanted'] > 0 or len(player.outstanding_trade_offer['property_set_wanted']) > 0):
            pass
        elif player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer['cash_offered'] > player.current_cash:
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
                net_amount_requested = -1*net_offer_worth
                count_create_new_monopoly = 0
                count_lose_existing_monopoly = 0
                for prop in player.outstanding_trade_offer['property_set_offered']:
                    if agent_helper_functions.will_property_complete_set(player,prop,current_gameboard):
                        count_create_new_monopoly += 1
                for prop in player.outstanding_trade_offer['property_set_wanted']:
                    if prop.color in player.full_color_sets_possessed:
                        count_lose_existing_monopoly += 1

                if count_lose_existing_monopoly - count_create_new_monopoly > 0:
                    reject_flag = 1
                elif count_lose_existing_monopoly - count_create_new_monopoly == 0:
                    if (player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer['cash_offered']) >= player.current_cash:
                        reject_flag = 1
                    elif player.current_cash - (player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer['cash_offered']) < current_gameboard['go_increment']/2:
                        reject_flag = 1
                    elif (player.current_cash - (player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer['cash_offered']) < current_gameboard['go_increment']) \
                            and net_offer_worth <= 0:
                        reject_flag =1
                    else:
                        reject_flag =0
                elif count_create_new_monopoly - count_lose_existing_monopoly > 0:
                    if (player.outstanding_trade_offer['cash_wanted'] - player.outstanding_trade_offer['cash_offered']) >= player.current_cash:
                        reject_flag = 1
                    else:
                        reject_flag = 0

            if reject_flag == 0:
                player.agent._agent_memory['previous_action'] = "accept_trade_offer"
                return ("accept_trade_offer", param)

    if "accept_sell_property_offer" in allowable_moves:
        param = dict()
        param['player'] = player.player_name
        param['current_gameboard'] = "current_gameboard"
        if player.outstanding_property_offer['asset'].is_mortgaged or player.outstanding_property_offer['price']>player.current_cash:
            pass
        elif player.current_cash-player.outstanding_property_offer['price'] >= current_gameboard['go_increment'] and \
            player.outstanding_property_offer['price']<=player.outstanding_property_offer['asset'].price:
            player.agent._agent_memory['previous_action'] = "accept_sell_property_offer"
            return ("accept_sell_property_offer", param)
        elif agent_helper_functions.will_property_complete_set(player, player.outstanding_property_offer['asset'],current_gameboard):
            if player.current_cash - player.outstanding_property_offer['price'] >= current_gameboard['go_increment']/2:
                player.agent._agent_memory['previous_action'] = "accept_sell_property_offer"
                return ("accept_sell_property_offer", param)

    if player.status != 'current_move':
        if "improve_property" in allowable_moves:
            param = agent_helper_functions.identify_improvement_opportunity(player, current_gameboard)
            if param:
                if player.agent._agent_memory['previous_action'] == "improve_property" and code == flag_config_dict['failure_code']:
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
                if player.current_cash-(m.mortgage*(1+current_gameboard['bank'].mortgage_percentage)) >= current_gameboard['go_increment'] and "free_mortgage" in allowable_moves:
                    param = dict()
                    param['player'] = player.player_name
                    param['asset'] = m.name
                    param['current_gameboard'] = "current_gameboard"
                    player.agent._agent_memory['previous_action'] = "free_mortgage"
                    return ("free_mortgage", param)

    else:
        if player.current_cash < current_gameboard['go_increment'] and "make_trade_offer" in allowable_moves:
            potential_offer_list = agent_helper_functions.identify_property_trade_offer_to_player(player, current_gameboard)
            potential_request_list = agent_helper_functions.identify_property_trade_wanted_from_player(player, current_gameboard)
            param_list = agent_helper_functions.curate_trade_offer_multiple_players(player, potential_offer_list, potential_request_list, current_gameboard, purpose_flag=1)
            if param_list and player.agent._agent_memory['previous_action'] != "make_trade_offer":
                return_action_list = []
                return_param_list = []
                if len(param_list)>1:
                    logger.debug(player.player_name + ": I am going to make multiple trade offers.")
                for param in param_list:
                    param['from_player'] = param['from_player'].player_name
                    param['to_player'] = param['to_player'].player_name
                    prop_set_offered = set()
                    for item in param['offer']['property_set_offered']:
                        prop_set_offered.add(item.name)
                    param['offer']['property_set_offered'] = prop_set_offered
                    prop_set_wanted = set()
                    for item in param['offer']['property_set_wanted']:
                        prop_set_wanted.add(item.name)
                    param['offer']['property_set_wanted'] = prop_set_wanted

                    player.agent._agent_memory['previous_action'] = "make_trade_offer"
                    return_action_list.append("make_trade_offer")
                    return_param_list.append(param)
                return (return_action_list, return_param_list)

        elif "make_trade_offer" in allowable_moves:
            potential_offer_list = agent_helper_functions.identify_property_trade_offer_to_player(player, current_gameboard)
            potential_request_list = agent_helper_functions.identify_property_trade_wanted_from_player(player, current_gameboard)
            param_list = agent_helper_functions.curate_trade_offer_multiple_players(player, potential_offer_list, potential_request_list, current_gameboard, purpose_flag=2)
            if param_list and player.agent._agent_memory['previous_action'] != "make_trade_offer":
                return_action_list = []
                return_param_list = []
                if len(param_list)>1:
                    logger.debug(player.player_name + ": I am going to make multiple trade offers.")
                for param in param_list:
                    param['from_player'] = param['from_player'].player_name
                    param['to_player'] = param['to_player'].player_name
                    prop_set_offered = set()
                    for item in param['offer']['property_set_offered']:
                        prop_set_offered.add(item.name)
                    param['offer']['property_set_offered'] = prop_set_offered
                    prop_set_wanted = set()
                    for item in param['offer']['property_set_wanted']:
                        prop_set_wanted.add(item.name)
                    param['offer']['property_set_wanted'] = prop_set_wanted
                    player.agent._agent_memory['previous_action'] = "make_trade_offer"
                    return_action_list.append("make_trade_offer")
                    return_param_list.append(param)
                return (return_action_list, return_param_list)

    if "skip_turn" in allowable_moves:
        player.agent._agent_memory['previous_action'] = "skip_turn"
        return ("skip_turn", dict())
    elif "concluded_actions" in allowable_moves:
        return ("concluded_actions", dict())
    else:
        logger.error("Exception")
        raise Exception


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
        logger.debug(player.player_name + ' has executed an unsuccessful postroll action, incrementing unsuccessful_tries ' +
                                          f'counter to {player.agent._agent_memory["count_unsuccessful_tries"]}')

    if player.agent._agent_memory['count_unsuccessful_tries'] >= UNSUCCESSFUL_LIMIT:
        logger.debug(player.player_name + ' has reached postroll unsuccessful action limits.')
        chosen_action = run_mcts(player, current_gameboard, allowable_moves)
        if chosen_action is not None:
            if chosen_action == "concluded_actions" and "concluded_actions" in allowable_moves:
                return ("concluded_actions", dict())
        if "concluded_actions" in allowable_moves:
            return ("concluded_actions", dict())
        else:
            logger.error("Exception")
            raise Exception

    current_location = current_gameboard['location_sequence'][player.current_position]
    chosen_action = run_mcts(player, current_gameboard, allowable_moves)
    if chosen_action is not None:
        if chosen_action == "buy_property" and "buy_property" in allowable_moves:
            params = dict()
            params['player'] = player.player_name
            params['asset'] = current_location.name
            params['current_gameboard'] = "current_gameboard"
            if make_buy_property_decision(player, current_gameboard, current_location):
                player.agent._agent_memory['previous_action'] = "buy_property"
                return ("buy_property", params)
            else:
                to_mortgage = agent_helper_functions.identify_potential_mortgage(player,current_location.price,True)
                if to_mortgage:
                    params['asset'] = to_mortgage.name
                    player.agent._agent_memory['previous_action'] = "mortgage_property"
                    return ("mortgage_property", params)
                else:
                    to_sell = agent_helper_functions.identify_potential_sale(player, current_gameboard, current_location.price,True)
                    if to_sell:
                        params['asset'] = to_sell.name
                        player.agent._agent_memory['previous_action'] = "sell_property"
                        return ("sell_property", params)

        if chosen_action == "concluded_actions" and "concluded_actions" in allowable_moves:
            return ("concluded_actions", dict())

    if "buy_property" in allowable_moves:
        if code == flag_config_dict['failure_code']:
            return ("concluded_actions", dict())

        params = dict()
        params['player'] = player.player_name
        params['asset'] = current_location.name
        params['current_gameboard'] = "current_gameboard"

        if make_buy_property_decision(player, current_gameboard, current_location):
            player.agent._agent_memory['previous_action'] = "buy_property"
            return ("buy_property", params)
        else:
            if agent_helper_functions.will_property_complete_set(player,current_location,current_gameboard):
                to_mortgage = agent_helper_functions.identify_potential_mortgage(player,current_location.price,True)
                if to_mortgage:
                    params['asset'] = to_mortgage.name
                    player.agent._agent_memory['previous_action'] = "mortgage_property"
                    return ("mortgage_property", params)
                else:
                    to_sell = agent_helper_functions.identify_potential_sale(player, current_gameboard, current_location.price,True)
                    if to_sell:
                        params['asset'] = to_sell.name
                        player.agent._agent_memory['previous_action'] = "sell_property"
                        return ("sell_property", params)

    if "concluded_actions" in allowable_moves:
        return ("concluded_actions", dict())
    else:
        logger.error("Exception")
        raise Exception


def make_buy_property_decision(player, current_gameboard, asset):
    decision = False
    if player.current_cash - asset.price >= current_gameboard['go_increment']:
        decision = True
    elif asset.price <= player.current_cash and agent_helper_functions.will_property_complete_set(player,asset,current_gameboard):
        decision = True
    return decision


def make_bid(player, current_gameboard, asset, current_bid):
    increment = (asset.price // 5) if asset.price > 0 else 1
    possible_bids = []
    for i in range(1,6):
        next_bid = current_bid + i * increment
        if next_bid < player.current_cash:
            possible_bids.append(next_bid)
    if not possible_bids:
        return 0

    bid_actions = set(str(b) for b in possible_bids)
    bid_actions.add("0")  # pass
    chosen_action = run_mcts(player, current_gameboard, bid_actions, max_iterations=30)
    if chosen_action is not None:
        if chosen_action == "0":
            return 0
        else:
            return int(float(chosen_action))

    # fallback heuristic
    if current_bid < asset.price:
        new_bid = current_bid + (asset.price-current_bid)//2
        if new_bid < player.current_cash:
            return new_bid
        else:
            return 0
    elif current_bid < player.current_cash and agent_helper_functions.will_property_complete_set(player,asset,current_gameboard):
        return current_bid+(player.current_cash-current_bid)//4
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
        elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
            continue
        else:
            mortgage_potentials.append((a,a.mortgage))
            max_sum += a.mortgage

    if mortgage_potentials and max_sum+player.current_cash >= 0:
        sorted_potentials = sorted(mortgage_potentials, key=lambda x: x[1])
        if len(sorted_potentials)>1:
            allowable = set("mortgage:"+p[0].name for p in sorted_potentials)
            chosen_action = run_mcts(player, current_gameboard, allowable, max_iterations=30)
            if chosen_action is not None:
                chosen_property = chosen_action.split(":")[1]
                params = dict()
                params['player'] = player.player_name
                params['asset'] = chosen_property
                params['current_gameboard'] = "current_gameboard"
                player.agent._agent_memory['previous_action'] = "mortgage_property"
                return ("mortgage_property", params)

        for p in sorted_potentials:
            if player.current_cash >= 0:
                return (None, flag_config_dict['successful_action'])
            params = dict()
            params['player'] = player.player_name
            params['asset'] = p[0].name
            params['current_gameboard'] = "current_gameboard"
            player.agent._agent_memory['previous_action'] = "mortgage_property"
            return ("mortgage_property", params)

    sale_potentials = []
    for a in sorted_player_assets_list:
        if a.color in player.full_color_sets_possessed:
            continue
        elif a.is_mortgaged:
            sale_potentials.append((a, (a.price*current_gameboard['bank'].property_sell_percentage)-((1+current_gameboard['bank'].mortgage_percentage)*a.mortgage)))
        elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
            continue
        else:
            sale_potentials.append((a,a.price*current_gameboard['bank'].property_sell_percentage))

    if sale_potentials:
        sorted_sp = sorted(sale_potentials, key=lambda x: x[1])
        if len(sorted_sp)>1:
            allowable = set("sell:"+p[0].name for p in sorted_sp)
            chosen_action = run_mcts(player, current_gameboard, allowable, max_iterations=30)
            if chosen_action is not None:
                chosen_property = chosen_action.split(":")[1]
                params = dict()
                params['player'] = player.player_name
                params['asset'] = chosen_property
                params['current_gameboard'] = "current_gameboard"
                player.agent._agent_memory['previous_action'] = "sell_property"
                return ("sell_property", params)

        for p in sorted_sp:
            if player.current_cash >= 0:
                return (None, flag_config_dict['successful_action'])
            params = dict()
            params['player'] = player.player_name
            params['asset'] = p[0].name
            params['current_gameboard'] = "current_gameboard"
            player.agent._agent_memory['previous_action'] = "sell_property"
            return ("sell_property", params)

    count = 0
    while (player.num_total_houses > 0 or player.num_total_hotels > 0) and count <3:
        count += 1
        sorted_assets_list = _set_to_sorted_list_assets(player.assets)
        sell_improvement_actions = []
        for a in sorted_assets_list:
            if a.loc_class == 'real_estate':
                if a.num_hotels > 0:
                    sell_improvement_actions.append(("sell_hotel:"+a.name,a))
                elif a.num_houses > 0:
                    sell_improvement_actions.append(("sell_house:"+a.name,a))

        if sell_improvement_actions:
            if len(sell_improvement_actions)>1:
                allowable = set(x[0] for x in sell_improvement_actions)
                chosen_action = run_mcts(player, current_gameboard, allowable, max_iterations=30)
                if chosen_action is not None:
                    act = chosen_action.split(":")[0]
                    prop_name = chosen_action.split(":")[1]
                    params = dict()
                    params['player'] = player.player_name
                    params['asset'] = prop_name
                    params['current_gameboard'] = "current_gameboard"
                    if act == "sell_house":
                        params['sell_house'] = True
                        params['sell_hotel'] = False
                    else:
                        params['sell_house'] = False
                        params['sell_hotel'] = True
                    player.agent._agent_memory['previous_action'] = "sell_house_hotel"
                    return ("sell_house_hotel", params)
            else:
                for sa in sell_improvement_actions:
                    if player.current_cash >= 0:
                        return (None, flag_config_dict['successful_action'])
                    act = sa[0].split(":")[0]
                    prop_name = sa[0].split(":")[1]
                    params = dict()
                    params['player'] = player.player_name
                    params['asset'] = prop_name
                    params['current_gameboard'] = "current_gameboard"
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
        if len(final_assets_list)>1:
            allowable = set("sell:"+a.name for a in final_assets_list)
            chosen_action = run_mcts(player, current_gameboard, allowable, max_iterations=30)
            if chosen_action is not None:
                chosen_property = chosen_action.split(":")[1]
                params = dict()
                params['player'] = player.player_name
                params['asset'] = chosen_property
                params['current_gameboard'] = "current_gameboard"
                player.agent._agent_memory['previous_action'] = "sell_property"
                return ("sell_property", params)
        for a in final_assets_list:
            if player.current_cash >= 0:
                return (None, flag_config_dict['successful_action'])
            params = dict()
            params['player'] = player.player_name
            params['asset'] = a.name
            params['current_gameboard'] = "current_gameboard"
            player.agent._agent_memory['previous_action'] = "sell_property"
            return ("sell_property", params)

    return (None, flag_config_dict['successful_action'])


def _set_to_sorted_list_mortgaged_assets(player_mortgaged_assets):
    player_m_assets_list = list()
    player_m_assets_dict = dict()
    for item in player_mortgaged_assets:
        player_m_assets_dict[item.name] = item
    for sorted_key in sorted(player_m_assets_dict):
        player_m_assets_list.append(player_m_assets_dict[sorted_key])
    return player_m_assets_list


def _set_to_sorted_list_assets(player_assets):
    player_assets_list = list()
    player_assets_dict = dict()
    for item in player_assets:
        player_assets_dict[item.name] = item
    for sorted_key in sorted(player_assets_dict):
        player_assets_list.append(player_assets_dict[sorted_key])
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
