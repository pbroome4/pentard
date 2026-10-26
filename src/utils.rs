use std::path::{Path, PathBuf};
use tracing::{trace, debug, info};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use burn::tensor::{Tensor,Device,TensorData};

use crate::gamestate::GameState;
use crate::tard::TrainingElement;
use crate::engine::SelfPlayElement;

#[derive(Debug)]
pub struct OutputManager{
    pub basepath_d: PathBuf,
    pub model_d: PathBuf,
    pub loss_f: PathBuf
}

impl OutputManager{
    pub fn new<P: Into<PathBuf>>(b: P, append: bool) -> io::Result<Self>{
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


pub fn selfplay_to_training<G:GameState>(sp_vec: Vec<SelfPlayElement<G>>, device: &Device) -> Vec<TrainingElement>{
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
