use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::io::{self, Write};
use burn::module::Module;
use tracing::{debug, info};
use tracing_subscriber::{fmt, prelude::*};
use tracing_subscriber::filter::{LevelFilter, EnvFilter};
use burn::tensor::{Tensor, Device, TensorData};
use burn::optim::{Adam, AdamConfig, GradientsParams};

mod tard;
mod gamestate;
mod evaluator;
mod engine;
mod pente;

use crate::gamestate::GameState;
use crate::pente::PenteState;
use crate::evaluator::*;
use crate::engine::SelfPlayElement;
use crate::tard::TrainingElement;
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
    let mut filter = EnvFilter::from_default_env()
        .add_directive(LevelFilter::WARN.into());
    filter = filter.add_directive("pentard=debug".parse().unwrap());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();
    train_tard();
}

fn train_tard(){
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
    let mut trn_buf = tard::TrainingBuffer::<50000>::new();
    let om = OutputManager::new("output", false).unwrap();

    let training_iterations = 20; // number of iterations of training per loop
    let training_size = 64;  // batch size for training.
    let n_rollouts = 200;
    let mut trn_iter_n: u32 = 0;
    let mut self_play_iter_n: u32 = 0;
    info!("Training batch size {}", training_size);
    loop{
        let mut self_play_states_added = 0;
        
        while self_play_states_added < 3*training_size{ // self play thread
            info!("Commencing self-play iteration {}", self_play_iter_n);
            let mut eval = TardEvaluator::new(tard.valid());
            let training_data = engine::run_self_play::<PenteState, TardEvaluator>(&mut eval, n_rollouts);
            let training_data =  selfplay_to_training(training_data, &device);
            info!("Adding {} training elements to buffer.", training_data.len());
            self_play_states_added += training_data.len();
            trn_buf.extend_front(training_data);
            self_play_iter_n += 1;
        }
        debug!("Training buffer size {}", trn_buf.len());
        for i in 0..training_iterations{     // training thread
            info!("Commencing training iteration {}", trn_iter_n);
            let trn_sample = trn_buf.sample(training_size);
            t_loss = tard::compute_loss(&tard, &trn_sample);
            info!("{}", t_loss);
            let grads = t_loss.loss.backward();
            let grad_params = GradientsParams::from_grads(grads, &tard);
            tard = opt.step(1e-4, tard, grad_params);
            trn_iter_n += 1;
        }

        if trn_iter_n % 20 == 0{
            info!("Saving model");
            let model_name = format!("model_{}", trn_iter_n);
            tard.clone().save_file(om.model_d.join(model_name))
                        .expect("Unable to save model");
            info!("Saving loss");
            let mut lf = OpenOptions::new()
                                  .write(true)
                                  .append(true)
                                  .open(&om.loss_f).unwrap();
            writeln!(lf,
                     "{}, {}, {}",
                     t_loss.loss.clone()
                                .into_scalar::<f32>(),
                     t_loss.cross_entropy_loss.clone()
                                              .into_scalar::<f32>(),
                     t_loss.mse_loss.clone()
                                    .into_scalar::<f32>(),
                    ).unwrap();
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
            info!("Deleting directory {}", basepath_d.display());
            fs::remove_dir_all(&basepath_d);
        }
        info!("Creating directory {}", basepath_d.display());
        fs::create_dir_all(&basepath_d)?;
        fs::create_dir_all(&model_d)?;
        let mut lf = OpenOptions::new().create(true)
                                  .append(true)
                                  .write(true)
                                  .open(&loss_f)?;
        lf.write(b"loss, cross_entropy_loss, mse_loss\n")?;

        let s = Self {
            basepath_d: basepath_d,
            model_d: model_d,
            loss_f: loss_f
        };
        info!("Created OutputManager: {:?}", s);
        return Ok(s);
    }
}


fn selfplay_to_training<G:GameState>(sp_vec: Vec<SelfPlayElement<G>>, device: &Device) -> Vec<TrainingElement>{
    let mut te_vec = Vec::with_capacity(sp_vec.len());
    for spe in sp_vec.into_iter(){
        let state_td = TensorData::new(spe.g.to_tensor(),
                                   [G::N_CHAN, G::ROWS, G::COLS]);
        let state = Tensor::from_data(state_td, device);
        let mcts_td = TensorData::new(spe.mcts_post, [G::ACTN_SPC_SIZE]);
        let mcts_post = Tensor::from_data(mcts_td, device);
        let outcome = Tensor::from_floats([spe.outcome], device);
        let te = TrainingElement{
            state: state,
            mcts_posterior: mcts_post,
            outcome: outcome
        };
        te_vec.push(te);
    }
    return te_vec;
}

