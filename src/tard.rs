use burn::prelude::*;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};
use burn::tensor::Device;
use burn::tensor::activation::{relu, tanh, log_softmax};
use tracing::{trace, debug, info};
use arraydeque::ArrayDeque;
use std::collections::VecDeque;
use rand::RngExt;


#[derive(Debug, Module)]
pub struct Tard{
    // 1. Shared Feature Extractor (Convolutional Trunk)
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    bn1: BatchNorm,
    bn2: BatchNorm,
    bn3: BatchNorm,

    // 2. Policy Head (Decides which move to make)
    // Output log probability of move to make
    policy_conv: Conv2d,
    policy_bn: BatchNorm,
    policy_fc: Linear,

    // 3. Value Head (Evaluates who is winning: -1 to +1)
    value_conv: Conv2d,
    value_bn: BatchNorm,
    value_fc1: Linear,
    value_fc2: Linear,
}
impl Tard {
    pub fn new(in_chan: usize, rows: usize, cols: usize, action_space_size:usize, device: &Device) -> Self {
        info!("Creating Tard in_chan:{}, rows: {},  cols:{}", in_chan, rows, cols);
        let conv1 = Conv2dConfig::new([in_chan, 64], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let conv2 = Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let conv3 = Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        let bn1 = BatchNormConfig::new(64).init(device);
        let bn2 = BatchNormConfig::new(64).init(device);
        let bn3 = BatchNormConfig::new(64).init(device);

        let policy_conv = Conv2dConfig::new([64, 2], [1, 1]).init(device);
        let policy_bn = BatchNormConfig::new(2).init(device);
        let policy_fc = LinearConfig::new(2 * rows * cols, rows * cols).init(device);

        let value_conv = Conv2dConfig::new([64, 1], [1, 1]).init(device);
        let value_bn = BatchNormConfig::new(1).init(device);
        let value_fc1 = LinearConfig::new(action_space_size, 64).init(device);
        let value_fc2 = LinearConfig::new(64, 1).init(device);

        let s = Self {
                conv1,
                conv2,
                conv3,
                bn1,
                bn2,
                bn3,
                policy_conv,
                policy_bn,
                policy_fc,
                value_conv,
                value_bn,
                value_fc1,
                value_fc2,
        };
        return s;
    }

    pub fn forward(&self, x: Tensor<4>) -> (Tensor<2>, Tensor<2>) {
        // 1. Convolutional Trunk
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = relu(x);

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = relu(x);

        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = relu(x);

        // 2. Policy Head
        let p = self.policy_conv.forward(x.clone());
        let p = self.policy_bn.forward(p);
        let p = relu(p);
        let [batch_size, channels, height, width] = p.shape().dims();
        let p = p.reshape([batch_size, channels * height * width]);
        let p = self.policy_fc.forward(p);
        let p = log_softmax(p, 1);

        // 3. Value Head
        let v = self.value_conv.forward(x);
        let v = self.value_bn.forward(v);
        let v = relu(v);
        let [batch_size, channels, height, width] = v.shape().dims();
        let v = v.reshape([batch_size, channels * height * width]);
        let v = self.value_fc1.forward(v);
        let v = self.value_fc2.forward(v);
        let v = tanh(v);

        (p, v)
    }
}

////// Training structures and Batch Utilities ////
#[derive(Clone)]
pub struct TrainingElement{
    pub state: Tensor<3>,       
    pub mcts_posterior: Tensor<1>,
    pub outcome: Tensor<1>,
}

pub struct TrainingBatch{
    pub state: Tensor<4>,
    pub mcts_posterior:Tensor<2>,
    pub outcome: Tensor<2>
}

impl TrainingBatch{
    pub fn from_training_elements(v: &Vec<TrainingElement>) -> TrainingBatch{
        let mut states = Vec::with_capacity(v.len());
        let mut mcts_posteriors = Vec::with_capacity(v.len());
        let mut outcomes = Vec::with_capacity(v.len());
        for te in v.iter(){
            states.push(te.state.clone());
            mcts_posteriors.push(te.mcts_posterior.clone());
            outcomes.push(te.outcome.clone());
        }
        let state_batch = Tensor::stack(states, 0);
        let mcts_posterior_batch = Tensor::stack(mcts_posteriors, 0);
        let outcome_batch = Tensor::stack(outcomes, 0);
        Self{
            state:state_batch,
            mcts_posterior:mcts_posterior_batch,
            outcome:outcome_batch
        }
    }
}


pub fn compute_loss(t: &Tard, v: &Vec<TrainingElement>) -> Tensor<1>{
    let batch = TrainingBatch::from_training_elements(v);
    let (tard_policy_batch, tard_value_batch) = t.forward(batch.state);
    
    // policy networkloss
    let mcts_posterior_probs = batch.mcts_posterior.exp();
    let cross_entropy_terms = mcts_posterior_probs * tard_policy_batch;
    let cross_entropy_loss = cross_entropy_terms.sum_dim(1)
                                                .neg()
                                                .mean();

    // value network loss
    let mse_loss = (tard_value_batch - batch.outcome).powf_scalar(2.0)
                                                     .mean();
    
    info!("CrossEntropyLoss: {}   MseLoss: {}", cross_entropy_loss, mse_loss);
    let loss = cross_entropy_loss + mse_loss;
    return loss;
}


pub struct TrainingBuffer<const CAP: usize>{
    memory_: VecDeque<TrainingElement>,
}

impl<const CAP: usize> TrainingBuffer<CAP> {
    pub fn new() -> Self {
        return Self{
            memory_: VecDeque::<TrainingElement>::with_capacity(CAP),
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

    pub fn extend_front<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = TrainingElement>,
    {
        return self.memory_.extend(iter);
    }

    pub fn len(&self) -> usize{
        return self.memory_.len();
    }
}

