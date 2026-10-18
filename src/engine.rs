use tracing::{debug, info};

use crate::tard;

pub trait GameState: Clone{
    // Get vector of legal actions 
    fn legal_actions(&self) -> Vec<usize>;

    // Get a random legal action (sampled uniformly)
    fn get_random_action(&self) -> Option<usize>;

    // Apply an action. action MUST be a legal action.
    fn apply_action(&mut self, action: &usize);

    // Is the game over
    fn is_terminal(&self) -> bool;

    // Get the terminal value
    fn get_terminal_value(&self) -> i32;     
}


pub fn run_self_play(t: &tard::Tard, batch_size:u32, n:u32){
    debug!("Running self play");
}
