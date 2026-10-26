
use std::collections::VecDeque;
use std::io::ErrorKind::TimedOut;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use burn::tensor::{Device, Tensor, TensorData};
use burn::module::Module;
use crossbeam::channel::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use tracing::{trace, debug, info, warn};

use crate::gamestate::*;
use crate::tard::*;

// Takes input state, runs it through network,
// returns dual head output
// vector is probability (NOT log) distribution over legal moves
// f32 is scalar score of current position
pub trait Evaluator <G: GameState>: Clone + Send + 'static{
    fn evaluate (&self, g: &G) -> (Vec<f32>, f32);
}

// Always returns uniform probability over legal moves and 0.0 score.
#[derive(Clone)]
pub struct NullEvaluator{}

impl <G:GameState> Evaluator<G> for NullEvaluator{
    fn evaluate (&self, g: &G) -> (Vec<f32>, f32) {
        let p = vec![1.0; G::ACTN_SPC_SIZE];
        return (legalize_policy(p, g), 0.0);
    }
}

// Runs input state through Tard and returns dual head output
#[derive(Clone)]
pub struct TardEvaluator{
    t: Tard
}

impl TardEvaluator{
    pub fn new(t: &Tard)->Self{
        return Self {t: t.valid()};
    }
}

impl <G:GameState> Evaluator<G> for TardEvaluator{
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

#[derive(Debug)]
struct BatchedEvalResponse{
    p: Vec<f32>,
    v: f32
}

#[derive(Debug)]
struct BatchedEvalRequest<G: GameState>{
    g: G,
    resp_chan: Sender<BatchedEvalResponse>    // one-shot response channel
}


// Will batch evaluations to Tard.
// Evaluations require synchronization with device (possibly gpu). 
// To reduce synchronization overhead, we prefer to batch Tard eval requests.
// This struct uses crossbeam channels to add evaluation requests to a common
// queue and return individual evaluation results.
#[derive(Clone)]
pub struct BatchedTardEvaluator <G:GameState>{
    s_eval: Sender<BatchedEvalRequest<G>>, // Channel to send batched requests to
    s_tard: Sender<Tard> // Channel to send updated Models
} 


impl <G: GameState> BatchedTardEvaluator<G>{
    pub fn new()->Self{
        let (s_eval, r_eval) = crossbeam::channel::unbounded::<BatchedEvalRequest<G>>();
        let (s_tard, r_tard) = crossbeam::channel::unbounded::<Tard>();
        start_bte_recieve_thread(r_eval, r_tard);
        let s = Self{
            s_eval: s_eval,
            s_tard: s_tard
        };
        return s;
    }
    
    pub fn init(self, t:Tard) -> Self{
        self.s_tard.send(t).unwrap();
        return self;
    }
}


impl <G:GameState> Evaluator<G> for BatchedTardEvaluator<G>{
    fn evaluate (&self, g: &G) -> (Vec<f32>, f32) {
        let (s,r) = crossbeam::channel::bounded::<BatchedEvalResponse>(1); // one-shot channel
        let req = BatchedEvalRequest{
            g: g.clone(),
            resp_chan: s,
        };
        self.s_eval.send(req).unwrap();
        let resp = r.recv().unwrap(); // blocking to revieve response
        return (resp.p, resp.v);
    }  
}

fn start_bte_recieve_thread<G: GameState> (
    r_eval: Receiver<BatchedEvalRequest<G>>,
    r_tard: Receiver<Tard>) -> JoinHandle<()>
{
    thread::spawn(move ||{  // Will run as long as there is a sender
        debug!("BTE reciever thread starting");
        debug!("BTE awaiting Tard");
        let mut is_running = true;
        let Ok(mut t) = r_tard.recv() else {
            debug!("BTE tard receive channel closed ... killing BTE thread");
            return;
        };
        debug!("BTE recieved Tard");
        const BATCH_SIZE: usize = 32;
        let mut q = VecDeque::<BatchedEvalRequest<G>>::new();
        while is_running{
            // Check for model updates
            let (t_updated, recv_connected) = check_for_model_update(r_tard.clone(), t);
            t = t_updated;
            is_running &= recv_connected;
            let device = t.devices()
                          .into_iter()
                          .next()
                          .expect("Expected at least one device");
            
            // Perform batch requests
            let batch_timeout = Duration::from_micros(10);
            let (n, recv_connected) = read_batch(&r_eval, &mut q, BATCH_SIZE, batch_timeout);
            is_running &= recv_connected;
            if q.len() > 0{
                let b_tens = batch_eval_tensor(&q, &device);
                let (p_batch, v_batch) = t.forward(b_tens);
                for i in 0..q.len(){
                    let p = p_batch.clone()
                                    .select_dim::<1>(0, i)
                                    .into_data()
                                    .try_into_vec::<f32>()
                                    .unwrap();
                    let v = v_batch.clone()
                                    .select_dim::<1>(0, i)
                                    .into_scalar::<f32>();
                    let p = legalize_policy(p, &q[i].g);
                    let resp = BatchedEvalResponse{
                        p: p,
                        v: v,
                    };
                    match q[i].resp_chan.send(resp) {
                        Ok(()) => {}
                        Err(x) => {
                            warn!("BTE eval response failed to send: {}", x);
                        }
                    }
                }
                q.clear();
            }
        }
        debug!("BatchedTardEvaluator receiver thread end");
    })
}


fn check_for_model_update(r_tard: Receiver<Tard>, t_in: Tard) -> (Tard, bool){
    let mut is_connected = true;
    let mut t = t_in;
    loop{
        match r_tard.try_recv(){
            Ok(t_updated) => {
                debug!("BTE updating model");
                t = t_updated;
            }
            Err(TryRecvError::Empty) => {
                break; // nothing to update
            }
            Err(TryRecvError::Disconnected) => {
                debug!("BTE tard receive channel closed ... signaling terminate for BTE thread");
                is_connected = false;
                break;
            }
        }
    }
    return (t, is_connected);
}

fn batch_eval_tensor <G:GameState>(
    q: &VecDeque<BatchedEvalRequest<G>>,
    device: &Device
) -> Tensor<4>{
    let batch_size = q.len();
    let mut b_vec: Vec<Tensor<3>> = Vec::with_capacity(batch_size);
    for i in 0..batch_size{
        let req = &q[i];
        let td = TensorData::new(req.g.to_tensor(),
                                 [G::N_CHAN, G::ROWS, G::COLS]);
        let t = Tensor::<3>::from_data(td, device);
        b_vec.push(t);
    }
    let b_tens: Tensor<4> = Tensor::stack(b_vec, 0);
    return b_tens;
}


// Blocking read requests off reciever into a queue.
// Blocking stops if any of these conditions is met:
//  1) length of the queue is/becomes >= batch_size
//  2) Timer exceeds timeout. Timer starts when len queue > 0.
//  3) receiver throws error (ex: disconnected)
// Returns 
//      Number of elements read into queue.
//      is_connected
fn read_batch<G:GameState>(
    r: &Receiver<BatchedEvalRequest<G>>,
    q: &mut VecDeque<BatchedEvalRequest<G>>,
    batch_size:usize,
    timeout: Duration) -> (usize, bool)
{
    let n = q.len();
    let mut is_connected = true;
    // Empty queue can just block read because no other requests are waiting
    if q.len() == 0{
        trace!("Queue is empty, performing blocking read"); 
        match r.recv(){
            Ok(req) => {
                q.push_back(req);
            }
            Err(x) => { // disconnected
                return (0, false);
            }
        }
    } // q.len() > 0, start timer
    let deadline = Instant::now() + timeout;
    while q.len() < batch_size{
        let now = Instant::now();
        if now > deadline{
            break;
        }
        let rx_timeout = deadline - now;
        match r.recv_timeout(rx_timeout) {
            Ok(req) => {
                q.push_back(req);
            }
            Err(RecvTimeoutError::Timeout) => {
                break;
            }
            Err(RecvTimeoutError::Disconnected) => {
                is_connected = false;
                break;
            } 
        } 
    }
    return (q.len() - n, is_connected);
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

