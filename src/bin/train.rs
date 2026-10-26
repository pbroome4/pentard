//use std::fs::{OpenOptions};
//use std::io::{Write};
//use burn::module::Module;
//use tracing::{debug, info};
//use tracing_subscriber::{prelude::*};
//use tracing_subscriber::filter::{LevelFilter, EnvFilter};
//use burn::tensor::{Device};
//use burn::optim::{AdamConfig, GradientsParams};
//
//
//
//use pentard::gamestate::GameState;
//use pentard::pente::PenteState;
//use pentard::evaluator::*;
//use pentard::utils::*;
//
//////////////////////////////////////////
//// Single threaded training implementation
//
fn main(){}
//    let mut filter = EnvFilter::from_default_env()
//        .add_directive(LevelFilter::WARN.into());
//    filter = filter.add_directive("pentard=debug".parse().unwrap());
//    tracing_subscriber::fmt()
//        .with_env_filter(filter)
//        .init();
//    train_tard();
//}
//
//fn train_tard(){
//    let device = Device::default().autodiff();
//    info!("Using Device: {:?}", device);
//    let mut tard = tard::Tard::new(PenteState::N_CHAN,
//        PenteState::ROWS,
//        PenteState::COLS,
//        PenteState::ACTN_SPC_SIZE,
//        &device);
//    debug!("{}", tard);
//    let training_batch_size = 256;
//    let n_rollouts = 256;
//    let mut t_loss: tard::TardLoss = Default::default();
//    let mut opt = AdamConfig::new().init();
//    let mut trn_buf = tard::TrainingBuffer::<50000>::new();
//    let om = OutputManager::new("output", false).unwrap();
//
//    let training_iterations = 20; // number of iterations of training per loop
//    let mut trn_iter_n: u32 = 0;
//    let mut self_play_iter_n: u32 = 0;
//    info!("Training batch size {}", training_size);
//    loop{
//        let mut self_play_states_added = 0;
//        //let mut eval = TardEvaluator::new(&tard);
//        let mut eval = BatchedTardEvaluator::new(&tard);
//        while self_play_states_added < 3*training_size{ // self play thread
//            info!("Commencing self-play iteration {}", self_play_iter_n);
//            let training_data = engine::run_self_play::<PenteState>(&mut eval, n_rollouts);
//            let training_data =  selfplay_to_training(training_data, &device);
//            info!("Adding {} training elements to buffer.", training_data.len());
//            self_play_states_added += training_data.len();
//            trn_buf.extend_front(training_data);
//            self_play_iter_n += 1;
//        }
//        debug!("Training buffer size {}", trn_buf.len());
//        for i in 0..training_iterations{     // training thread
//            info!("Commencing training iteration {}", trn_iter_n);
//            let trn_sample = trn_buf.sample(training_size);
//            t_loss = tard::compute_loss(&tard, &trn_sample);
//            info!("{}", t_loss);
//            let grads = t_loss.loss.backward();
//            let grad_params = GradientsParams::from_grads(grads, &tard);
//            tard = opt.step(1e-4, tard, grad_params);
//            trn_iter_n += 1;
//        }
//
//        if trn_iter_n % 20 == 0{
//            info!("Saving model");
//            let model_name = format!("model_{}", trn_iter_n);
//            tard.clone().save_file(om.model_d.join(model_name))
//                        .expect("Unable to save model");
//            info!("Saving loss");
//            let mut lf = OpenOptions::new()
//                                  .write(true)
//                                  .append(true)
//                                  .open(&om.loss_f).unwrap();
//            writeln!(lf,
//                     "{}, {}, {}",
//                     t_loss.loss.clone()
//                                .into_scalar::<f32>(),
//                     t_loss.cross_entropy_loss.clone()
//                                              .into_scalar::<f32>(),
//                     t_loss.mse_loss.clone()
//                                    .into_scalar::<f32>(),
//                    ).unwrap();
//        }
//    }
//}
