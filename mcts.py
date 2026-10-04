import numpy as np
from typing import List
import random

import engine


class MCTSNode:
    """ 
    Holds values for single node in a monte carlo tree search (MCTS) tree.
    """
    def __init__(self, actions: np.ndarray, num_visits = 0, parent=None):
        """ Actions is list of the allowable actions from the current game state represented in this node."""
        self.parent = parent
        self.actions_count = dict.fromkeys(actions, 0)
        self.children = {}  # mapping action_id -> child node 
        self.num_visits = num_visits # Note: is >=0. Is the number of times backprop has run on the node. 
        self.val_avg = 0


    def select_ucb_action(self, C=np.sqrt(2)) -> int:
        action_val = {} # mapping between action_id and ucb metric value
        actions = list(self.actions_count.keys())
        random.shuffle(actions) # shuffle the list of actions so that any unexplored node is sampled randomly.
        for action in actions:
            if action not in self.children:
                return action   # unexplored node, ucb metric inf, return.
            child = self.children[action]
            ucb = child.val_avg + C * np.sqrt(np.log(self.num_visits) / child.num_visits)
            action_val[action] = ucb
        ucb_action = max(action_val, key=action_val.get)    # Gets key associated with max value in the dict.
        return ucb_action
    

    def back_propogate(self, reward):
        print(reward)
        if self.num_visits == 0:
            self.val_avg = reward
        else:   # self.num_visits > 0
            cum_value = self.val_avg * (self.num_visits) + reward
            self.val_avg = cum_value / (self.num_visits + 1)
        self.num_visits += 1

        if self.parent is not None:
            self.parent.back_propogate(reward)

    def _count_children(self):
        n = 0
        for child in self.children.values():
            n += 1 + child._count_children()
        return n


    def __str__(self):
        action, count = max(self.actions_count.items(), key=lambda item: item[1])    # grab (key,value) with highest value
        row, col = engine.action_id_2_coord(action)
        return f"num_vists: {self.num_visits}    val_avg: {self.val_avg}   rec:{(row,col)}   num_children:{self._count_children()}"



    
def random_rollout(state:engine.GameState, depth_limit = None):
    """
    Repeated chooses legal moves uniformly at random until hitting a terminal state or a depth limit. Returns the resulting state.
    Warning, not setting a depth limit risk an infinite loop if no terminal state is reach.
    """
    if depth_limit is None:
        depth_limit = np.inf
    depth = 0 
    while(depth < depth_limit):
        if state.is_terminal():
            break
        legal_actions = arg_mask(state.get_legal_moves())
        action = legal_actions[np.random.randint(0, len(legal_actions))]
        state.apply_action(action)
    return state


def mcts_sim(root_node: MCTSNode, state: engine.GameState):
    """
    Perform one simulation of pure monte carlo tree search(MCTS) using upper confidence bound(UCB).
    We start from the root, repeatedly selecting actions and traversing down the tree. Eventually we hit a leaf. From here, we perform
    perform a "random rollout" until we hit a terminal state. The first action we took beyond the leaf node gets added to our tree once
    it has been visited twice. This is tracked by MCTSNode.actions_count.
    
    state: the current game state. This value is assumed muteable. Consider passing in a copy of the state to prevent unexpected state changes
    
    Notes:
    Uses delayed node expansion, only creating a node if it has been visited once already. This reduces memory of the tree.
    """
    plyr = state.is_white()
    node = root_node
    while True:
        action = node.select_ucb_action()
        state.apply_action(action)
        node.actions_count[action] += 1
        if state.is_terminal():
            break
        if node.actions_count[action] > 1 and action not in node.children: # delayed node expansion
            legal_actions = arg_mask(state.get_legal_moves())
            node.children[action] = MCTSNode(legal_actions, parent=node)
        if action in node.children:    
            node = node.children[action]
            continue
        else: # node.actions_count[action] == 1
            state = random_rollout(state, depth_limit=200)
            break
    # state is either terminal, or hit a depth limit.
    if state.is_terminal():
        reward = state.get_terminal_value(plyr)
    else:
        reward = 0 # hit a depth limit. No reward or punishment.
    node.back_propogate(reward)


def mcts(state, n = 2000):
    legal_actions = arg_mask(state.get_legal_moves())
    root_node = MCTSNode(legal_actions)
    for i in range(n):
        copy_state = state.clone()
        mcts_sim(root_node, copy_state)
    print(root_node)


def arg_mask(mask)->List:
    return np.where(mask == 1)[0]


if __name__ == "__main__":
    brd = engine.GameState()
    while(True):
        print(brd)
        player_str = {True: "White", False:"Black"}
        user_in = input(f"Enter {player_str[brd.is_white()]} move as row, col: ")
        coord = user_in.split(",")
        if len(coord) == 2:
            row = int(coord[0])
            col = int(coord[1])
            if brd.play_safe(row, col):
                if brd.is_terminal():
                    print(f"Game over. {player_str[brd.is_white()]} Win!")
                    break
                mcts(brd)
            else:
                print("Invalid move, try again")
                continue

        else:
            print("Unable to parse input, try again")
            continue
            