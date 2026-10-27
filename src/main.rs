use std::fs::{OpenOptions};
use std::io::{Write};
use std::thread;
use burn::module::Module;
use pentard::tard::Tard;
use tracing::{debug, info};
use tracing_subscriber::{prelude::*};
use tracing_subscriber::filter::{LevelFilter, EnvFilter};
use burn::tensor::{Device};
use burn::optim::{AdamConfig, GradientsParams};
use crossbeam::channel::{Sender, Receiver};

mod tard;
mod gamestate;
mod evaluator;
mod engine;
mod pente;
mod utils;

use crate::gamestate::GameState;
use crate::pente::PenteState;
use crate::engine::*;
use crate::evaluator::*;
use crate::utils::*;
////////////////////////////////////////
//      ## Main thread
//  1) Initialize model, vars, logging, etc  
//  2) Repeatedly self-play and train model
//      2.1) Spawn eval and worker threads using current model
//      2.2) Collect worker data
//      2.3) Perform training
//      2.4) periodically update model in the eval thread.
//  3) Eventually terminate
//
//       #### Worker thread (many threads) ###
//   1) Create game at starting position (start with batch size 1)
//   2) Run MCTS
//       2.1) For each game, Run N(=800, 1600 etc?) MCTS runouts
//            This MCTS requires network prior and value output for PUCT
//            which requires a eval thread interaction
//       2.2) At the end of N MCTS runouts, make best move found for the game. 
//       2.3) repeat until game ends
//       2.4) Return training data IE:
//            For each state in each game, return (state, win/loss/draw),
//            network value, actual game terminal value(win/loss).
//   3) Send training data to a replay buffer in main thread
//
//      #### Evaluator thread ####
//  1) Recieve network eval requests on a queue
//  2) Batch up requests and send to network
//  3) Get evaluations and redistribute.
//  4) Occassionaly resync with main thread.


fn main(){
    let mut filter = EnvFilter::from_default_env()
        .add_directive(LevelFilter::WARN.into());
    filter = filter.add_directive("pentard=trace".parse().unwrap());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();
    train_tard();
}

fn train_tard(){
    // Configs
    let n_cpus = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1); // Fallback to 1 if it cannot be determined
    let n_worker_threads = 5;
    let n_rollouts = 200;
    let n_elements_per_train = 1;
    let training_steps_per_iter = 10;
    let batch_size = 128;
    let train_mem_capacity:usize = 30000;

    let om = OutputManager::new("output", false).unwrap();
    let device = Device::default().autodiff();
    info!("Using Device: {:?}", device);
    let mut tard = tard::Tard::new(PenteState::N_CHAN,
        PenteState::ROWS,
        PenteState::COLS,
        PenteState::ACTN_SPC_SIZE,
        &device);
    debug!("{}", tard);
    let mut t_loss: tard::TardLoss = Default::default();
    let mut opt = AdamConfig::new().init();
    let mut trn_buf = tard::TrainingBuffer::new(train_mem_capacity);

    //let eval = BatchedTardEvaluator::new().init(tard); //BatchedTardEvaluator::new().init(tard);
    let eval = TardEvaluator::new(&tard);
    let (s, r) = crossbeam::channel::unbounded();
    let handles = spawn_workers::<PenteState>(n_worker_threads, s, eval, n_rollouts);       

    loop{
        let mut count = 0;
        while let Ok(req) = r.recv(){
            debug!("Reading game element #{}", count);
            count+=1;
        }
        drop(r); // should cause workers to fail to send game and terminate
        for h in handles.into_iter(){
            h.join().unwrap();
        }
        break;
    }
}


fn spawn_workers <G: GameState>(
    n_workers: usize,
    s: Sender<SelfPlayElement<G>>,
    eval: impl Evaluator<G>,
    n_rollouts: usize) -> Vec<thread::JoinHandle<()>>
{
    let mut v = Vec::new(); 
    for i in 0..n_workers{
        let s_clone = s.clone();
        let eval_clone = eval.clone();
        info!("Spawning worker thread #{}", i);
        let h = thread::spawn(move ||{
            let game_data = run_self_play::<G>(&eval_clone, n_rollouts);
            debug!("Worker thread #{} Self play complete.", i);
            debug!("Worker thread #{} adding {} training elements", i, game_data.len());
            for d in game_data.into_iter(){
                s_clone.send(d).unwrap();   
            }
            debug!("Closing worker thread #{}", i);
        });
        v.push(h);
    }
    return v;
}
