use burn::module::Module;
use tracing::{Level, debug, info};
use tracing_subscriber;
use burn::tensor::{Tensor, Device};
use burn::optim::{Adam, AdamConfig, GradientsParams};

mod tard;
mod engine;
mod pente;

use pente::PenteState;
use engine::GameState;
////////////////////////////////////////
//       ### Thread 1 ####
//   1) Sample training data from replay buffer
//   2) update network
//
//       #### Thread 2 ###
//   1) Create batch of games at starting position (start with batch size 1)
//   2) Call library to run MCTS
//       2.1) For each game, Run N(=800, 1600 etc?) MCTS runouts
//            This MCTS requires network prior and value output for PUCT
//       2.2) At the end of N MCTS runouts, make best move found for the game. 
//       2.3) repeat until all games end
//       2.4) Return training data IE:
//            For each state in each game, return (state, win/loss/draw),
//            network value, actual game terminal value(win/loss).
//   3) Add training data to a replay buffer
//   4) Resync with thread1 model which has had aditional training
//      


fn main(){
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG) // Set max level: TRACE, DEBUG, INFO, WARN, ERROR, or OFF
        .init();
    train_tard();
}

fn train_tard(){
    let device = Device::default();
    info!("Using Device: {:?}", device);
    let mut tard = tard::Tard::new(PenteState::N_CHAN,
        PenteState::ROWS,
        PenteState::COLS,
        PenteState::ACTN_SPC_SIZE,
        &device);
    debug!("{}", tard);
    let mut opt = AdamConfig::new().init();
    let mut trn_buf = tard::TrainingBuffer::<10000>::new();

    let mut trn_iter_n: u32 = 0;
    let mut self_play_iter_n: u32 = 0;
    let training_size = 256;
    let n_rollouts = 800;
    info!("Training batch size {}", training_size);
    loop{
        let mut self_play_states_added = 0;
        let inference_tard = tard.valid();
        while self_play_states_added < 3*training_size{
            info!("Commencing self-play iteration {}", self_play_iter_n);
            let training_data = engine::run_self_play::<PenteState>(&inference_tard, n_rollouts);
            info!("Added {} training elements to buffer", training_data.len());
            self_play_states_added += training_data.len();
            trn_buf.extend_front(training_data);
            self_play_iter_n += 1;
            
        }
        debug!("Training buffer size {}", trn_buf.len());
        for i in 0..20{
            info!("Commencing training iteration {}", trn_iter_n);
            let trn_sample = trn_buf.sample(training_size);
            let loss = tard::compute_loss(&tard, &trn_sample);
            info!("Loss: {}", loss);
            let grads = loss.backward();
            let grad_params = GradientsParams::from_grads(grads, &tard);
            tard = opt.step(1e-4, tard, grad_params);
            trn_iter_n += 1;
        }
    }
}


