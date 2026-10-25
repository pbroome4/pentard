use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::io::{self, Write};
use burn::backend::autodiff::checkpoint::base;
use burn::module::Module;
use burn::nn::loss;
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
//       ### Thread A, (one thread) ####
//   1) Sample training data from replay buffer
//   2) update network
//
//       #### Thread B (many threads) ###
//   1) Create game at starting position (start with batch size 1)
//   2) Run MCTS
//       2.1) For each game, Run N(=800, 1600 etc?) MCTS runouts
//            This MCTS requires network prior and value output for PUCT
//            which requires a thread C interaction
//       2.2) At the end of N MCTS runouts, make best move found for the game. 
//       2.3) repeat until all games end
//       2.4) Return training data IE:
//            For each state in each game, return (state, win/loss/draw),
//            network value, actual game terminal value(win/loss).
//   3) Add training data to a replay buffer in thread type A
//   4) Resync with thread type A model which has had aditional training
//      
//      #### Thread C (one thread) ####
//  1) Recieve network eval requests on a queue
//  2) Batch up requests and send to network
//  3) Get evaluations and redistribute.


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
    let om = OutputManager::new("output").unwrap();

    let mut trn_iter_n: u32 = 0;
    let mut self_play_iter_n: u32 = 0;
    let training_size = 256;
    let n_rollouts = 400;
    info!("Training batch size {}", training_size);
    loop{
        let mut self_play_states_added = 0;
        let inference_tard = tard.valid();
        while self_play_states_added < 3*training_size{ // self play thread
            info!("Commencing self-play iteration {}", self_play_iter_n);
            let training_data = engine::run_self_play::<PenteState>(&inference_tard, n_rollouts);
            info!("Adding {} training elements to buffer.", training_data.len());
            self_play_states_added += training_data.len();
            trn_buf.extend_front(training_data);
            self_play_iter_n += 1;
        }
        debug!("Training buffer size {}", trn_buf.len());
        for i in 0..20{     // training thread
            info!("Commencing training iteration {}", trn_iter_n);
            let trn_sample = trn_buf.sample(training_size);
            let loss = tard::compute_loss(&tard, &trn_sample);
            info!("Loss: {}", loss);
            let grads = loss.backward();
            let grad_params = GradientsParams::from_grads(grads, &tard);
            tard = opt.step(1e-4, tard, grad_params);
            trn_iter_n += 1;
        }

        if trn_iter_n % 20 == 0{
            info!("Saving model");
            let model_name = format!("model_{}", trn_iter_n);
            tard.clone().save_file(om.model_d.join(model_name))
                        .expect("Unable to save model");
        }
    }
}

#[derive(Debug)]
struct OutputManager{
    basepath_d: PathBuf,
    model_d: PathBuf,
    loss_f: PathBuf
}

impl OutputManager{
    fn new<P: Into<PathBuf>>(b: P, append: bool) -> io::Result<Self>{
        let basepath_d = b.into();
        let model_d = basepath_d.join("models");
        let loss_f = basepath_d.join("loss.csv");

        if append == false{
            fs::remove_dir_all(&basepath_d)?;
        }
        fs::create_dir_all(&basepath_d)?;
        fs::create_dir_all(&model_d)?;
        let mut lf_opt = OpenOptions::new().create(true)
                                  .append(true)
                                  .write(true)
                                  .open(&loss_f)?;
        l.write(b"loss, cross_entropy_loss, mse_loss\n")?;

        let s = Self {
            basepath_d: basepath_d,
            model_d: model_d,
            loss_f: loss_f
        };
        info!("Created OutputManager: {:?}", s);
        return Ok(s);
    }
}




