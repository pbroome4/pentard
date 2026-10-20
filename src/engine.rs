use tch::{Kind, Tensor};
use rand::prelude::*;
use tracing::{trace, debug, info};
use std::{fmt};

use crate::tard::{self, TrainingElement};

type Action = usize;
pub trait GameState: Clone + fmt::Display{
    const ACTN_SPC_SIZE: usize;
    const N_TENSOR_CHANNELS: usize;

    fn new() -> Self;

    fn to_tensor(&self) -> Tensor;

    // usize tensor of 0's and 1's. 0=illegal. 1=legal
    fn legal_actions_mask(&self) -> Tensor;

    // Get vector of legal actions 
    //fn legal_actions(&self) -> Vec<usize>;

    // Generate a random legal action (sampled uniformly)
    // Must only be called in non-terminal state.
    // Calling in terminal state is UB.
    fn get_random_action(&self) -> Action;

    // Apply an action. action MUST be a legal action.
    // Calling with an illegal action is UB.
    fn apply_action(&mut self, action: Action);

    // Is the game over
    fn is_terminal(&self) -> bool;

    // Returns:
    //  +1 on current player win
    //   0 on Draw
    //  -1 on current player loss.
    // Note: for games like pente/chess, current player will only ever draw/lose.
    fn get_terminal_value(&self) -> i32;

    // Returns:
    //  +1 on Player 0 win
    //   0 on Draw
    //  -1 on Player 1 win
    // TODO DEPRECATE
    fn get_objective_terminal_value(&self) -> i32;     
}

// Plan was to run multiple games in parallel so we could batch gpu operations.
// Think it is still possible to batch gpu operations even simulating games one at a time, so might remove batch_size
// For no batch_size is assumed equal to 1.
pub fn run_self_play <G: GameState>(t: &tard::Tard, batch_size: u32, num_sims: u32) -> Vec<tard::TrainingElement>{
    debug!("Running self play");
    let mut te_vec = Vec::<tard::TrainingElement>::new();
    let mut state = G::new();
    let mut move_n = 0;
    while !state.is_terminal(){
        debug!("Analzying move {}", move_n);
        let mcts_posterior = mcts(&state, t, num_sims);
        debug!("{}", mcts_posterior);
        debug!("\n{}", state);
        let action = i64::try_from(mcts_posterior.multinomial(1, false)).unwrap() as usize;
        debug!("Playing action: {}", action);
        let te = TrainingElement{
            state: state.to_tensor(),
            mcts_posterior: mcts_posterior,
            outcome: 0.0,
        };
        te_vec.push(te);
        state.apply_action(action);
        move_n += 1;
    }
    let outcome = state.get_terminal_value() as f32;
    let mut polarity: f32 = 1.0;
    for te in te_vec.iter_mut().rev(){
        te.outcome = outcome * polarity;
        polarity *= -1.0;
    }
    return te_vec;
}
// Run MCTS from the current state. State must be non-terminal. num sims >= 1
pub fn mcts <GS: GameState> (state: &GS, t: &tard::Tard, num_sims: u32) -> Tensor{
    // Run mcts and return the evaluated mcts policy
    let mut tree: MctsTree = MctsTree::new();
    for i in 0..num_sims{
        mcts_sim(&mut tree, state, t);
    }
    let root = tree.get(0).unwrap();
    debug!("Root node: {}", root);
    return mcts_tree_posterior(&tree, 0, GS::ACTN_SPC_SIZE);
}

// Performs a single MCTS simulation using PUCT
fn mcts_sim(tree: &mut MctsTree, state_orig: &impl GameState, t: &tard::Tard){
    let mut state = state_orig.clone();
    let mut node_id = 0; // Root
    loop{ // Drill down to a leaf node.
        if let Some(node) = tree.get(node_id){
            if node.is_expanded(){
                let (child_id, action) = tree.puct_select(node_id);
                node_id = child_id;
                state.apply_action(action);
            }else{
                break;
            }
        }
    }
    // Is leaf node (possibly terminal)
    let node = tree.get(node_id).unwrap();
    let value = 
        if state.is_terminal(){
            state.get_terminal_value() as f32
        }else{
            tree.expand(node_id, &state, t)
        };
    tree.back_propogate(node_id, value); 
}

// Returns log mcts posterior probability distributions
fn mcts_tree_posterior(tree: &MctsTree, node_id: NodeId, actn_spc_size: usize) -> Tensor{
    let node = tree.get(node_id).unwrap();
    let mut child_id = node.first_child.unwrap(); // state assumed non-terminal and sims >= 1
    let mut p = vec![0.0; actn_spc_size];
    loop{
        let child = tree.get(child_id).unwrap();
        let action = child.action.unwrap();
        p[action] = (child.visits as f32) / (node.visits as f32);
        if let Some(next_child_id) = child.next_sibling{
            child_id = next_child_id;
        }else{
            break;
        }
    }
     // normalize (because parent visits = 1+sum(children))
     // and convert to logits
    let mut t = Tensor::from_slice(&p);
    let t_sum = t.sum(Kind::Float);
    t = t / t_sum;
    t = t.log();
    return t;
}



type NodeId = usize;

pub struct MctsNode {
    pub parent: Option<NodeId>,       // Parent index for backpropagation. None for root node.
    pub first_child: Option<NodeId>,  // First element in the children linked list.
                                       // None if node is terminal or not yet expanded.
    pub next_sibling: Option<NodeId>, // Next sibling index.
    pub action: Option<Action>,       // The action that led to this state
    pub prior: f32,                   // The prior probability of selecting this node from the parent.                 
    pub visits: u32,                  // Number of times this node has been visited. 1 + Sum(child_visits)
    pub value: f32,                   // Total value for this node.
                                      // If not expanded, is 0.0.
                                      // If expanded, value - sum(child_values).
    //pub is_terminal: bool,            // If this node is a terminal game state.            
}

impl MctsNode{
    pub fn is_expanded(&self) -> bool{
        return self.first_child != None; 
    }
}

impl fmt::Display for MctsNode{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut avg_val = 0.0;
        if self.visits > 0{
            avg_val = self.value / (self.visits as f32);
        }
        write!(f, "visits: {}   avg_value: {}   is_expanded: {}", self.visits, avg_val, self.is_expanded())?;
        return Ok(());
    }
}



// Memory-pool implementation of MCTS Tree
// Root node has NodeId == 0;
pub struct MctsTree {
    pub nodes_: Vec<MctsNode>,
}

impl MctsTree{
    pub fn new() -> Self{
        let mut x = Self{
            nodes_: Vec::new(),
        };
        let root = MctsNode{
            parent: None,
            first_child: None,     
            next_sibling: None,
            action:  None,
            prior: 1.0,
            visits: 0,
            value: 0.0,
        };
        x.nodes_.push(root);
        return x;
    }
    
    pub fn get(&self, node_id: NodeId) -> Option<&MctsNode>{
        return self.nodes_.get(node_id);
    }

    pub fn get_mut(&mut self, node_id: NodeId) -> Option<&mut MctsNode>{
        return self.nodes_.get_mut(node_id);
    }

    
    fn back_propogate(&mut self, node_id: NodeId, value: f32){
        let mut curr_id = node_id;
        let mut polarity: f32 = 1.0;
        while let Some(curr) = self.get_mut(curr_id){
            curr.value += value * polarity;
            curr.visits += 1;
            polarity *= -1.0;
            if let Some(parent_id) = curr.parent{
                curr_id = parent_id;
            }else{
                break;
            }
        }
    }

    // Returns probabilistic upper confidence bound applied to trees (PUCT)
    // and returns best option.
    // Only call on node_id that have been expanded.
    // Calling on node_id that are not expanded is UB.
    fn puct_select(&self, node_id:NodeId) -> (NodeId, Action){
        let curr = self.get(node_id).unwrap();
        let mut child_id = curr.first_child.unwrap();
        let mut best_child_id = child_id;
        let mut best_score = self.calc_puct(child_id);
        loop{
            if let Some(child) = self.get(child_id){
                if let Some(next_child_id) = child.next_sibling{
                    child_id = next_child_id;
                    let score = self.calc_puct(child_id);
                    if score > best_score{
                        best_child_id = child_id;
                        best_score = score;
                    }
                    continue;
                } 
            }
            break;
        }
        let action = self.get(best_child_id).unwrap().action.unwrap();
        return (best_child_id, action);
    }

    // Calculates PUCT score for a given node.
    // Node must have a parent. 
    fn calc_puct(&self, node_id: NodeId) -> f32{
        let c = 1.5;    // c hyperparameter. Typically 1.0 to 2.5. More complex models will adjust this over time.
        let node = self.get(node_id).unwrap();
        let parent = self.get(node.parent.unwrap()).unwrap();
        let exploitation_term =
            if node.visits == 0{
                0.0
            }else{
                node.value / (node.visits as f32)
            };
        let exploration_term = c * node.prior * (parent.visits as f32).sqrt() / ((1 + node.visits) as f32);
        return exploitation_term + exploration_term;
    }
    

    // Only call on non-terminal states. Caling on terminal state is UB.
    // Returns the value of the node
    fn expand(&mut self, node_id:NodeId, state: &impl GameState, t: &tard::Tard) -> f32{
        debug_assert_eq!(state.is_terminal(), false);
        trace!("Expanding node {}", node_id);
        let state_tens = state.to_tensor();
        let in_tens = Tensor::stack(&[state_tens], 0);  // Make a batchsize of 1
        let (p_batch, v_batch) = t.forward_t(&in_tens, false);
        let p = p_batch.flatten(0, -1);
        let v = v_batch.flatten(0, -1);
        let val = f32::try_from(v)
                          .expect("Unable to convert value tensor to scalar");
        let legal_p = legalize_policy_tensor(&p, state);                
        let legal_p_vec = Vec::<f32>::try_from(legal_p.exp())  // Convert from log probability to probability
                                     .expect("Unable to convert policy Tensor to vector");
        for action in 0..legal_p_vec.len(){
            if legal_p_vec[action] > 0.0{   // Only add legal actions.
                self.add_child(action, node_id, legal_p_vec[action]);
            }
        }
        return val;
    }

    // Create a new node in the arena-pool. Return it's id.
    fn add_child(&mut self, action: Action, parent: NodeId, prior:f32) -> NodeId{
        let child_id: NodeId = self.nodes_.len();
        let child = MctsNode{
            action: Some(action),
            parent: Some(parent),
            first_child: None,
            next_sibling: self.nodes_[parent].first_child,
            prior: prior,
            visits: 0,
            value: 0.0,
        };
        self.nodes_.push(child);
        self.nodes_[parent].first_child = Some(child_id);
        trace!("Added child {}", child_id);
        return child_id;
    }
}

// p is log probability
// Output Tensor is log probability
fn legalize_policy_tensor(p: &Tensor, state: &impl GameState) -> Tensor{
    let legal_mask = state.legal_actions_mask();
    let mut legalized = p + (legal_mask - 1) * 1e9;  // Makes log probabilities extremely negative so softmax results in probability 0.
    legalized = legalized.log_softmax(-1, Kind::Float);
    return legalized;
}
