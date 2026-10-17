use tch::IValue::Device;
use tch::nn::{self, BatchNormConfig, LinearConfig, OptimizerConfig, linear};
use tch::Tensor;
use tracing::info;

use crate::pente;


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

#[derive(Debug)]
struct Tard{
    // 1. Shared Feature Extractor (Convolutional Trunk)
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    conv3: nn::Conv2D,
    bn1: nn::BatchNorm,
    bn2: nn::BatchNorm,
    bn3: nn::BatchNorm,

    // 2. Policy Head (Decides which move to make)
    policy_conv: nn::Conv2D,
    policy_bn: nn::BatchNorm,
    policy_fc: nn::Linear,

    // 3. Value Head (Evaluates who is winning: -1 to +1)
    value_conv: nn::Conv2D,
    value_bn: nn::BatchNorm,
    value_fc1: nn::Linear,
    value_fc2: nn::Linear,
}

impl Tard{
    pub fn new(vs: &nn::Path, in_chan: i64)->Self{
        
        let c = nn::ConvConfig{
            stride: 1,
            padding: 1,
            ..Default::default() // it implements default. .. is rust spread
        };
        let b = BatchNormConfig{..Default::default()};
        let l = LinearConfig{..Default::default()};

        // Input is shape (batch_size, input_channels, board_width, board_height)
        let conv1 = nn::conv2d(vs / "conv1", in_chan, 64, 3, c);
        let conv2 = nn::conv2d(vs / "conv2", in_chan, 64, 3, c);
        let conv3 = nn::conv2d(vs / "conv3", in_chan, 64, 3, c);
        let bn1 = nn::batch_norm2d(vs / "bn1", 64, b);
        let bn2 = nn::batch_norm2d(vs / "bn2", 64, b);
        let bn3 = nn::batch_norm2d(vs / "bn3", 64, b);
        
        let policy_conv = nn::conv2d(vs / "policy_conv", 64, 2, 1, c);
        let policy_bn = nn::batch_norm2d(vs / "policy_bn", 2, b);
        let policy_fc = nn::linear(vs / "policy_fc", 2*81, 81, l);
        
        let value_conv = nn::conv2d(vs / "value_conv", 64, 1, 3, c);
        let value_bn = nn::batch_norm2d(vs / "value_bn", 1, b);
        let value_fc1 = nn::linear(vs / "value_fc1", 81, 64, l);
        let value_fc2 = nn::linear(vs / "value_fc2",64, 1, l);
        return Self{conv1, conv2, conv3,
                    bn1, bn2, bn3,
                    policy_conv, policy_bn, policy_fc,
                    value_conv, value_bn, value_fc1, value_fc2};
    }
}

//Basically impl nn::ModuleT except i want to return pair of Tensors
impl Tard{
    fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor,Tensor){
        let x  = xs.apply(&self.conv1)
                   .apply_t(&self.bn1, train)
                   .relu()
                   .apply(&self.conv2)
                   .apply_t(&self.bn2, train)
                   .relu()
                   .apply(&self.conv3)
                   .apply_t(&self.bn3, train)
                   .relu();
        let p = x.apply(&self.policy_conv)
                 .apply_t(&self.policy_bn, train)
                 .relu()
                 .apply_t(&self.policy_fc, train);
        let v = x.apply(&self.value_conv)
                 .apply_t(&self.value_bn, train)
                 .relu()
                 .apply(&self.value_fc1)
                 .apply(&self.value_fc2);
        return (p,v)
    }
}

pub fn train_tard(){
    let device = tch::Device::cuda_if_available();
    match device {
        tch::Device::Cpu => info!("CUDA is not available. Falling back to CPU."),
        tch::Device::Cuda(id) => info!("CUDA is available! Using GPU device ID: {}", id),
        _ => println!("Using device: {:?}", device),
    }
    let vs = nn::VarStore::new(device);
    let tard = Tard::new(&vs.root(), 3);
    let mut optimizer = nn::adam(0.9, 0.99, 0.0).build(&vs, 1e-4).unwrap();

    let mut iteration_n:u32 = 0;
    while true{
        iteration_n += 1;
        info!("Iteration {}", iteration_n);
        let _gaurd = tch::no_grad(||{
            
        });
    }
}
