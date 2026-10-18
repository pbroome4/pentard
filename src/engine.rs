use tch::{Kind, Tensor, kind};
use rand::prelude::*;
use tracing::{trace, debug, info};
use std::fmt;

use crate::tard;


pub trait GameState: Clone{
    fn new() -> Self;

    fn to_tensor(&self) -> Tensor;

    fn action_space_size(&self) -> usize;

    fn legal_actions_mask(&self) -> Tensor;

    // Get vector of legal actions 
    //fn legal_actions(&self) -> Vec<usize>;

    // Generate a random legal action (sampled uniformly)
    // Must only be called in non-terminal state.
    // Calling in terminal state is UB.
    fn get_random_action(&self) -> usize;

    // Apply an action. action MUST be a legal action.
    // Calling with an illegal action is UB.
    fn apply_action(&mut self, action: usize);

    // Is the game over
    fn is_terminal(&self) -> bool;

    // Get the terminal value.
    // Returns:
    //  +1 on Player 0 win
    //   0 on Draw
    //  -1 on Player 1 win
    fn get_terminal_value(&self) -> i32;     
}

// Plan was to run multiple games in parallel so we could batch gpu operations.
// Think it is still possible to batch gpu operations even simulating games one at a time, so might remove batch_size
pub fn run_self_play <G: GameState>(t: &tard::Tard, batch_size:u32, n:u32) -> Vec<tard::TrainingElement>{
    debug!("Running self play");
    let mut te_vec = Vec::<tard::TrainingElement>::new();
    let mut g = G::new();
    let mut move_n = 0;
    while !g.is_terminal(){
        debug!("Move {}", move_n);
        
        let state_tens = g.to_tensor();
        let in_tens = Tensor::stack(&[state_tens], 0);
        let (policy, value) = t.forward_t(&in_tens, false);
        debug!("Policy: {}\nValue:{}", policy, value);
        
        let leg_acts_mask = g.legal_actions_mask();
        let num_leg_acts = leg_acts_mask.sum(Kind::Float);   // Should be >= 0 since non-terminal
        let te = tard::TrainingElement{
            state: g.to_tensor(),
            mcts_policy:  &leg_acts_mask / &num_leg_acts,
            outcome: 0,
        };
        te_vec.push(te);
        g.apply_action(g.get_random_action());
        move_n += 1;
    }
    let outcome = g.get_terminal_value();
    let mut polarity = 1;
    for te in te_vec.iter_mut(){
        te.outcome = outcome * polarity;
        polarity *= -1;
    }
    return te_vec;
}


type NodeId = usize;

pub struct MctsNode<A> {
    pub action_: Option<A>,            // The move that led to this state
    pub parent_: Option<NodeId>,       // Parent index for backpropagation
    pub first_child_: Option<NodeId>,  // First element in the sibling list
    pub next_sibling_: Option<NodeId>, // Next sibling index
    pub visits_: u32,                  // Counter
    pub total_value_: f32,             // Accumulated evaluation value
    pub is_expanded_: bool,
}

impl<A> fmt::Display for MctsNode<A>{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut avg_val = 0.0;
        if self.visits_ > 0{
            avg_val = self.total_value_ / (self.visits_ as f32);
        }
        write!(f, "visits: {}   avg_value: {}   is_expanded: {}", self.visits_, avg_val, self.is_expanded_)?;
        return Ok(());
    }
}

// Memory-pool implementation of MCTS Tree
// Root node has NodeId == 0;
pub struct MctsTree<A> {
    pub nodes_: Vec<MctsNode<A>>,
}

impl<A> MctsTree<A>{
    pub fn new() -> Self{
        let mut x = Self{
            nodes_: Vec::new(),
        };
        let root = MctsNode{
            action_:  None,
            parent_: None,
            first_child_: None,     
            next_sibling_: None,
            visits_: 0,
            total_value_: 0.0,
            is_expanded_: false,
        };
        x.nodes_.push(root);
        return x;
    }

}


