use std::sync::mpsc::channel;

use burn::tensor::{Tensor, Device, TensorData};
use burn::module::Module;
use crossbeam::channel::{self, Sender};

use crate::gamestate::*;
use crate::tard::*;

// Takes input state, runs in through network,
// returns dual head output
// vector is probability distribution over legal moves
// f32 is scalar score of current position
trait Evaluate <G: GameState>{
    fn evaluate (&self, g: &G) -> (Vec<f32>, f32);
}

// Always returns uniform probability over legal moves and 0.0 score.
struct NullEvaluator{}

impl <G:GameState> Evaluate<G> for NullEvaluator{
    fn evaluate (&self, g: &G) -> (Vec<f32>, f32) {
        let p = vec![1.0; G::ACTN_SPC_SIZE];
        return (legalize_policy(p, g), 0.0);
    }
}

// Runs input state through Tard and returns dual head output
struct TardEvaluator{
    t: Tard
}

impl <G:GameState> Evaluate<G> for TardEvaluator{
    fn evaluate (&self, g:&G) -> (Vec<f32>, f32) {
        let device = self.t.devices()
                  .into_iter()
                  .next()
                  .expect("Expected at least one device");
        let state_tens_data = TensorData::new(g.to_tensor(),
                                 [G::N_CHAN, G::ROWS, G::COLS]);
        let state_tens = Tensor::<3>::from_data(state_tens_data, &device);
        let state_batch_tens = Tensor::stack(vec![state_tens], 0);  // Make a batchsize of 1
        let (p_batch, v_batch) = self.t.forward(state_batch_tens);
        let p_tens: Tensor<1> = p_batch.flatten(0, -1); // debatch
        let v_tens = v_batch.flatten::<1>(0, -1); // debatch
        let p: Vec<f32> = p_tens.exp().into_data().try_into_vec().unwrap();
        let v = v_tens.into_scalar::<f32>();
        let p = legalize_policy(p, g);
        return (p, v);
    }
}


struct BatchedEvalResponse{
    p: Vec<f32>,
    v: f32
}

struct BatchedEvalRequest<G: GameState>{
    g: G,
    resp_chan: Sender<BatchedEvalResponse>    // one-shot respone channel
}


// Will batch evaluations to Tard.
// Evaluations require synchronization with device (possibly gpu). 
// To reduce synchronization overhead, we prefer to batch Tard eval requests.
// This struct uses crossbeam channels to add evaluation requests to a common
// queue and return individual evaluation result.
struct BatchedTardEvaluator <G:GameState>{
    t: Tard,
    s: Sender<BatchedEvalRequest<G>>
} 

impl <G:GameState> Evaluate<G> for BatchedTardEvaluator<G>{
    fn evaluate (&self, g: &G) -> (Vec<f32>, f32) {
        let (s,r) = channel::bounded::<BatchedEvalResponse>(1); // one-shot channel
        let device = self.t.devices()
                  .into_iter()
                  .next()
                  .expect("Expected at least one device");
        let td = TensorData::new(g.to_tensor(),
                    [G::N_CHAN, G::ROWS, G::COLS]);
        let t = Tensor::<3>::from_data(td, &device);
        let req = BatchedEvalRequest{
            g: g.clone(),
            resp_chan: s,
        };
        let resp = r.recv().unwrap(); // blocking to revieve response
        return (resp.p, resp.v);
    }  
}



// p is probability vector
// Output is probability vector
fn legalize_policy(mut p: Vec<f32>, g: &impl GameState) -> Vec<f32>{
    let legal_mask = g.legal_actions_mask();
    for i in 0..p.len(){
        if legal_mask[i] == 0.0{
            p[i] = 0.0;
        }
    }
    return normalize(p);
}


// normalizes probability vector to 1.0.
// Assumes sim is non-zero and that each element >=0
fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let sum:f32 = v.iter().sum();
    for e in v.iter_mut(){
        *e = *e/sum;
    }
    return v;
}

