import numpy as np
from typing import List

import engine


class MCTSNode:
    """ 
    Holds values for single node in a monte carlo tree search (MCTS) tree.
    """
    def __init__(self, actions: np.ndarray, num_visits = 1, parent=None):
        """ Actions is list of the allowable actions from the current game state represented in this node."""
        self.parent = parent
        self.actions_count = dict.fromkeys(actions, 0)
        self.children = {}  # mapping action_id -> child node 
        self.num_vists = num_visits # Note: is >=1 and >= sum(actions_count). 
                                    # Is greater than the sum when MCTS is performing delayed node expansion.
        self.val_avg = 0


    def select_ucb_action(self) -> int:
        return 


    def back_propogate(self, reward):
        if self.num_vists == 0:
            self.val_avg = reward
        else:   # self.num_visits > 0
            cum_value = self.val_avg * (self.num_vists - 1) + reward
            self.val_avg = cum_value / self.num_vists
        
    
def random_rollout(state, depth_limit = None):
    """
    Repeated chooses legal moves uniformly at random until hitting a terminal state or a depth limit. Returns the resulting state.
    Warning, not setting a depth limit risk an infinite loop if no terminal state is reach.
    """

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
        node.num_visits += 1
        if state.is_terminal():
            break
        if node.actions_count[action] >= 1 and action not in node.children: # delayed node expansion
            legal_actions = np.where(state.get_legal_moves() == 1)[0]
            node.children[action] = MCTSNode(legal_actions, num_visits=2, parent=node)
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



def mcts_puct():
    """ Uses predictor upper confidence bounds applied to trees (puct) for monte carlo tree search(MCTS) """
    pass
