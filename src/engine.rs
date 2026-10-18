use tracing::{debug, info};
use tch::{Kind, Tensor, kind};

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


pub fn run_self_play <G: GameState>(t: &tard::Tard, batch_size:u32, n:u32) -> Vec<tard::TrainingElement>{
    debug!("Running self play");
    let mut te_vec = Vec::<tard::TrainingElement>::new();
    let mut g = G::new();
    let mut move_n = 0;
    while !g.is_terminal(){
        debug!("Game 0 move {}", move_n);
        let leg_acts_mask = g.legal_actions_mask();
        let num_leg_acts = leg_acts_mask.sum(Kind::Float);   // Should be >= 0 since non-terminal
        let te = tard::TrainingElement{
            state: g.to_tensor(),
            mcts_policy:  &leg_acts_mask / &num_leg_acts,
            outcome: 0,
        };
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
