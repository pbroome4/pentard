use burn::module::Module;
use burn::tensor::{Tensor, TensorData};
use burn::tensor::activation::{log_softmax};
use tracing::{trace, debug, info};
use std::{fmt};

use crate::tard::{self, TrainingElement};

type Action = usize;


// GameState is an abstraction of a 2D board game.
pub trait GameState: Clone + fmt::Display{
    const ACTN_SPC_SIZE: usize; // Input action space is scalar [0, ACTN_SPC_SIZE).
    const N_CHAN: usize; // Game can be represented as flattened tensor [[[f32; COLS]; ROWS]; N_CHAN]
    const ROWS: usize;
    const COLS: usize;

    fn new() -> Self;

    fn to_tensor(&self) -> Vec<f32>; // flattened N_CHAN x ROWS x COLS

    // usize tensor of 0's and 1's. 0.0=illegal. 1.0=legal
    fn legal_actions_mask(&self) -> Vec<f32>;

    // Generate a random legal action (sampled uniformly)
    // Must only be called in non-terminal state.
    // Calling in terminal state is UB.
    //fn get_random_action(&self) -> Action;

    // Apply an action. action MUST be a legal action.
    // Calling with an illegal action is UB.
    fn apply_action(&mut self, action: Action);

    // Is the game over
    fn is_terminal(&self) -> bool;

    // Returns:
    //  +1 on Player 0 win
    //   0 on Draw
    //  -1 on Player 1 win
    fn get_terminal_value(&self) -> i32;     
}

pub fn run_self_play <G: GameState>(
    t: &tard::Tard,
    num_sims: u32,
) -> Vec<tard::TrainingElement>{
    debug!("Running self play");
    let device = t.devices()
                  .into_iter()
                  .next()
                  .expect("Expected at least one device");
    let mut te_vec = Vec::new();
    let mut state = G::new();
    let mut move_n = 0;
    while !state.is_terminal(){
        debug!("Current State: \n{}", state);
        debug!("Computing self-play move#: {}", move_n);
        let (mcts_pri, mcts_post) = mcts(&state, t, num_sims);
        debug!("MCTS prior logits: {:?}", mcts_pri);
        debug!("MCTS posterior logits:{:?}", mcts_post);
        let mcts_tens_d = TensorData::new(mcts_post, [G::ACTN_SPC_SIZE]);
        let mcts_tens = Tensor::from_floats(mcts_tens_d, &device);
        let action = mcts_tens.clone()
                              .exp()
                              .categorical(1)
                              .into_scalar::<u32>() as usize;
        debug!("Playing action: {}", action);
        let td = TensorData::new(state.to_tensor(),
                                 [G::N_CHAN, G::ROWS, G::COLS]);
        let te = TrainingElement{
            state: Tensor::from_data(td, &device),
            mcts_posterior: mcts_tens,
            outcome: Tensor::from_floats([0.0_f32], &device)
        };
        te_vec.push(te);
        state.apply_action(action);
        move_n += 1;
    }
    let outcome = state.get_terminal_value() as f32;
    for te in te_vec.iter_mut(){
        te.outcome = Tensor::from_floats([outcome], &device);
    }
    return te_vec;
}


// Run MCTS from the current state. State must be non-terminal. num sims >= 1.
// Returns log probabiltiy posterior.
pub fn mcts <G: GameState> (state: &G, t: &tard::Tard, num_sims: u32) -> (Vec<f32>, Vec<f32>){
    // Run mcts and return the evaluated mcts policy
    let mut tree: MctsTree = MctsTree::new(1.0);
    for _ in 0..num_sims{
        mcts_sim(&mut tree, state, t);
    }
    let root = tree.get(0).unwrap();
    debug!("Mcts root node: {}", root);
    let mcts_pri = mcts_tree_prior(&tree, 0, G::ACTN_SPC_SIZE);
    let mcts_post = mcts_tree_posterior(&tree, 0, G::ACTN_SPC_SIZE);
    trace!("total probs should be 1.0. mcts_pri: {}  mcts_post: {}", total_prob(&mcts_pri), total_prob(&mcts_post));
    return (mcts_pri, mcts_post);
}

// Performs a single MCTS simulation using PUCT
fn mcts_sim <G:GameState>
(
    tree: &mut MctsTree,
    state_orig: &G,
    t: &tard::Tard
){
    let mut state = state_orig.clone();
    let mut node_id = 0; // Root
    loop{ // Drill down to a leaf node.
        let node = tree.get(node_id).unwrap();
        if node.is_expanded(){
            let (child_id, action) = tree.puct_select(node_id);
            node_id = child_id;
            state.apply_action(action);
        }else{
            break;
        }
    }
    // Is leaf node (possibly terminal)
    let value = 
        if state.is_terminal(){
            state.get_terminal_value() as f32
        }else{
            tree.expand(node_id, &state, t)
        };
    tree.back_propogate(node_id, value); 
}


// Returns log mcts posterior probability distributions
// node_id must be a non-terminal and non-leaf node.
fn mcts_tree_posterior(tree: &MctsTree, node_id: NodeId, actn_spc_size: usize) -> Vec<f32>{
    let node = tree.get(node_id).unwrap();
    let mut p = vec![0.0; actn_spc_size];
    for child_id in tree.children(node_id){
        let child = tree.get(child_id).unwrap();
        let action = child.action.unwrap();
        p[action] = (child.visits as f32) / (node.visits as f32);
    }
    let p_sum: f32 = p.iter().sum();
    for e in p.iter_mut(){
        *e = (*e / p_sum).ln();
    }
    return p;
}

// Returns log probability
fn mcts_tree_prior(tree: &MctsTree, node_id: NodeId, actn_spc_size: usize) -> Vec<f32>{
    let mut p = vec![f32::NEG_INFINITY; actn_spc_size];
    for child_id in tree.children(node_id){
        let child = tree.get(child_id).unwrap();
        let action = child.action.unwrap();
        p[action] = child.prior.ln();
    }
    return p;
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

pub struct Children<'a>{
    tree: &'a MctsTree,
    next_id: Option<NodeId>
}

impl <'a>Iterator for Children<'a>{
    type Item = NodeId;
    fn next(&mut self) -> Option<Self::Item> {
        let current_id = self.next_id?;
        self.next_id = self.tree.get(current_id)?.next_sibling;
        return Some(current_id);
    }
}


// Memory-pool implementation of MCTS Tree
// Root node has NodeId == 0;
pub struct MctsTree {
    nodes_: Vec<MctsNode>,
    c_puct: f32,
}

impl MctsTree{
    pub fn new(c_puct: f32) -> Self{
        let mut x = Self{
            nodes_: Vec::new(),
            c_puct: c_puct // c hyperparameter. Typically 1.0 to 2.5. More complex models will adjust this over time.
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


    fn children(&self, node_id: NodeId) -> Children<'_>{
        let first_child = self.get(node_id).unwrap().first_child;
        return Children { tree: &self, next_id: first_child }
    }
    
    fn back_propogate(&mut self, node_id: NodeId, value: f32){
        let mut curr_id = node_id;
        while let Some(curr) = self.get_mut(curr_id){
            curr.value += value;
            curr.visits += 1;
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
        let mut best_child_id = usize::MAX;
        let mut best_score = f32::NAN;
        for child_id in self.children(node_id){
            let score = self.calc_puct(child_id);
            if best_score.is_nan() || score > best_score{
                best_child_id = child_id;
                best_score = score;
            }
        }
        let action = self.get(best_child_id).unwrap().action.unwrap();
        return (best_child_id, action);
    }

    // Calculates PUCT score for a given node.
    // Node must have a parent. 
    fn calc_puct(&self, node_id: NodeId) -> f32{
        let node = self.get(node_id).unwrap();
        let parent = self.get(node.parent.unwrap()).unwrap();
        let exploitation_term =
            if node.visits == 0{
                0.0
            }else{
                let polarity = if self.depth(node_id)%2==0 {1.0} else {-1.0};
                // polarity determines if we are minimizing or maximizing player
                polarity * node.value / (node.visits as f32)
            };
        let exploration_term = self.c_puct * node.prior *
                              (parent.visits as f32).sqrt() / ((1 + node.visits) as f32);
        return exploitation_term + exploration_term;
    }
    

    // Only call on non-terminal states. Caling on terminal state is UB.
    // Returns the value of the node
    fn expand <G:GameState>(&mut self,
        node_id:NodeId,
        state: &G,
        t: &tard::Tard
    ) -> f32{
        debug_assert_eq!(state.is_terminal(), false);
        trace!("Expanding node {}", node_id);
        let device = t.devices()
                  .into_iter()
                  .next()
                  .expect("Expected at least one device");
        let state_tens_data = TensorData::new(state.to_tensor(),
                                 [G::N_CHAN, G::ROWS, G::COLS]);
        let state_tens = Tensor::<3>::from_data(state_tens_data, &device);
        let in_tens = Tensor::stack(vec![state_tens], 0);  // Make a batchsize of 1
        let (p_batch, v_batch) = t.forward(in_tens);
        let p: Tensor<1> = p_batch.flatten(0, -1); // debatch
        let v = v_batch.flatten::<1>(0, -1); // debatch
        let val = v.into_scalar::<f32>();
        let probs:Vec<f32> = p.exp().into_data().try_into_vec().unwrap();
        let legal_probs = legalize_policy(probs, state);
        for action in 0..legal_probs.len(){
            if legal_probs[action] > 0.0{   // Only add legal actions.
                self.add_child(action, node_id, legal_probs[action]);
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

    fn depth(&self, node_id:NodeId) -> usize {
        let mut count = 0;
        let mut curr_id = node_id;
        while let Some(next_id) = self.get(curr_id).unwrap().parent{
            curr_id = next_id;
            count += 1;
        } 
        return count;
    }
}

// p is probability vector
// Output is probability vector
fn legalize_policy(mut p: Vec<f32>, state: &impl GameState) -> Vec<f32>{
    let legal_mask = state.legal_actions_mask();
    for i in 0..p.len(){
        if legal_mask[i] == 0.0{
            p[i] = 0.0;
        }
    }
    return normalize(p);
}

// normalizes probability vector to 1.0.
// Assumes sim is non-zero and that each element >=0
fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let sum:f32 = v.iter().sum();
    for e in v.iter_mut(){
        *e = *e/sum;
    }
    return v;
}

// p is potentailly unormalized log probabilities.
// Returns the sum(exp(p));
fn total_prob(p: &Vec<f32>)->f32{
    let mut sum = 0.0;
    for f in p{
        sum += f.exp();
    }
    return sum;
}