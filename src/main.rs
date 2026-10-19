use tracing::{Level, debug, info};
use tracing_subscriber;
use tch::IValue::Device;
use tch::nn::{self, BatchNormConfig, LinearConfig, OptimizerConfig, linear};
use tch::Tensor;

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
    let device = tch::Device::cuda_if_available();
    match device {
        tch::Device::Cpu => info!("CUDA is not available. Falling back to CPU."),
        tch::Device::Cuda(id) => info!("CUDA is available! Using GPU device ID: {}", id),
        _ => println!("Using device: {:?}", device),
    }
    let vs = nn::VarStore::new(device);
    let tard = tard::Tard::new(&vs.root(),
                               PenteState::N_TENSOR_CHANNELS as i64,
                               pente::ROWS as i64,
                               pente::COLS as i64);
    let mut opt = nn::adam(0.9, 0.99, 0.0).build(&vs, 3e-4).unwrap();
    let mut trn_buf = tard::TrainingBuffer::<50000>::new();

    let mut iter_n: u32 = 0;
    loop{
        info!("Iteration {}", iter_n);
        info!("Commencing self play");
        let _gaurd = tch::no_grad(||{
            let training_data = engine::run_self_play::<PenteState>(&tard, 1, 200);
            trn_buf.extend_front(training_data);
            debug!("TrainingBuffer size: {}", trn_buf.len());
        });
        
        info!("Commencing training");
        let training_size = 256;
        let trn_sample = trn_buf.sample(training_size);
        opt.zero_grad();
        let loss = tard::compute_loss(&tard, &trn_sample);
        info!("Loss: {}", loss);
        loss.backward();
        opt.step();

        iter_n += 1;

    }
}


