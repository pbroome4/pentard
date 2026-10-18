use tch::nn::{self, BatchNormConfig, ConvConfig, LinearConfig, linear};
use tch::Tensor;
use tracing::{trace, debug, info};
use arraydeque::ArrayDeque;
use rand::RngExt;
use std::default;
use std::ops::{Deref, DerefMut};


#[derive(Debug)]
pub struct Tard{
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
        let conv2 = nn::conv2d(vs / "conv2", 64, 64, 3, c);
        let conv3 = nn::conv2d(vs / "conv3", 64, 64, 3, c);
        let bn1 = nn::batch_norm2d(vs / "bn1", 64, b);
        let bn2 = nn::batch_norm2d(vs / "bn2", 64, b);
        let bn3 = nn::batch_norm2d(vs / "bn3", 64, b);
        

        let policy_conv = nn::conv2d(vs / "policy_conv", 64, 2, 1, ConvConfig::default());
        let policy_bn = nn::batch_norm2d(vs / "policy_bn", 2, b);
        let policy_fc = nn::linear(vs / "policy_fc", 2*81, 81, l);
        
        let value_conv = nn::conv2d(vs / "value_conv", 64, 1, 1, ConvConfig::default());
        let value_bn = nn::batch_norm2d(vs / "value_bn", 1, b);
        let value_fc1 = nn::linear(vs / "value_fc1", 81, 64, l);
        let value_fc2 = nn::linear(vs / "value_fc2",64, 1, l);
        return Self{conv1, conv2, conv3,
                    bn1, bn2, bn3,
                    policy_conv, policy_bn, policy_fc,
                    value_conv, value_bn, value_fc1, value_fc2};
    }

    //Basically impl nn::ModuleT except i want to return pair of Tensors
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor,Tensor){
        trace!("forward_t() with xs = {}  train = {}", xs, train);
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
                 .flatten(1, -1)
                 .apply(&self.policy_fc);
        let v = x.apply(&self.value_conv)
                 .apply_t(&self.value_bn, train)
                 .relu()
                 .flatten(1, -1)
                 .apply(&self.value_fc1)
                 .apply(&self.value_fc2);
        return (p,v)
    }
}

pub struct TrainingElement{
    pub state: tch::Tensor,       
    pub mcts_policy: tch::Tensor,
    pub outcome: i32,
}

impl Clone for TrainingElement{
    fn clone(&self) -> Self {
        return Self {
            state: self.state.copy(),
            mcts_policy: self.mcts_policy.copy(),
            outcome: self.outcome
        };
    }
}

pub struct TrainingBuffer<const CAP: usize>{
    memory_: ArrayDeque<TrainingElement, CAP>,
}

impl<const CAP: usize> TrainingBuffer<CAP> {
    pub fn new() -> Self {
        return Self{
            memory_: ArrayDeque::<TrainingElement, CAP>::new(),
        };
    }

    pub fn sample(&self, size: usize) -> Vec<TrainingElement> {
        let mut v = Vec::<TrainingElement>::new();
        let l = self.memory_.len();
        if l > 0{
            for _ in 0..size {
                let i = rand::rng().random_range(0..l);                
                v.push(self.memory_[i].clone());
            }
        }
        return v;
    }
}

// Transparently expose all other ArrayDeque methods
impl <const CAP: usize> Deref for TrainingBuffer<CAP>{
    type Target = ArrayDeque<TrainingElement, CAP>;
    fn deref(&self) -> &Self::Target {
        &self.memory_
    }
}

impl <const CAP: usize> DerefMut for TrainingBuffer<CAP>{
    fn deref_mut(&mut self) -> &mut Self::Target {
        return &mut self.memory_;
    }
}