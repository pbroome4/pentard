use std::collections::VecDeque;
use std::thread::{self, JoinHandle};
use burn::backend::ir::NumericOperationIr::Sum;
use burn::tensor::{Device, Tensor, TensorData};
use burn::module::Module;
use crossbeam::channel::{Receiver, RecvError, RecvTimeoutError, Sender, TryRecvError};
use tracing::{trace, debug, info, warn};
use std::marker::PhantomData;

use crate::gamestate::*;
use crate::tard::*;
use crate::utils::*;
////////////////////////////////////////////////////////////////////////
//
//                    Evaluator
//
///////////////////////////////////////////////////////////////////////
// Takes input state, evaluates it, and returns (policy, value) dual output.
// vector is a probability (NOT log) distribution over legal moves
// f32 is scalar score of current position.
pub trait Evaluator <G: GameState>: 'static + Send + Clone{
    fn evaluate (&self, g: &G) -> (Vec<f32>, f32);
}

///////////////////////////////////////////////////////////////////////
//
//                      Serverless Evaluator
//
///////////////////////////////////////////////////////////////////////
// Used in single threaded application. Evaluators are updated by simply
// creating new intances


// Always returns uniform probability over legal moves and 0.0 score.
#[derive(Clone)]
pub struct NullEvaluator{}

impl <G> Evaluator<G> for NullEvaluator
where 
    G:GameState
{
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
    fn evaluate (&self, g: &G) -> (Vec<f32>, f32) {
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
////////////////////////////////////////////////////////////////////////////
//
//                     Server Based Evaluator
//
//////////////////////////////////////////////////////////////////////////
// Covers the centralized and decentralized evaluators

pub trait Server<G: GameState>{
    type ClientType: Evaluator<G>;
    fn get_client(&self) -> Self::ClientType;
}


//////////////////////////////////////////////////////////////////////////////
//
//                      Centralized Server Evaluators
//
//////////////////////////////////////////////////////////////////////////////
// This architecture features clients which request evaluations from a
// centralized server. Evaluator is updated by sending an update to the
// server

#[derive(Debug, Clone)]
pub struct CEvalClient <G>{

    s_eval: Sender<EvalRequest<G>>
}

impl <G:GameState> Evaluator<G> for CEvalClient<G>

{
    fn evaluate(&self, g: &G) -> (Vec<f32>, f32){
        let (s,r) = crossbeam::channel::unbounded();
        let req = EvalRequest{
            g: g.clone(),
            resp_chan: s
        };
        self.s_eval.send(req).unwrap();
        let resp = r.recv().unwrap();
        return (resp.p, resp.v);
    }
}


pub struct CNullEvalServer<G>{
    s_eval:  Sender<EvalRequest<G>>
}

impl <G: GameState> Server<G> for CNullEvalServer<G>{
    type ClientType = CEvalClient<G>;

    fn get_client(&self) -> CEvalClient<G>{
        return CEvalClient{s_eval: self.s_eval.clone()};
    }
}


impl <G:GameState> Default for CNullEvalServer<G>{
    fn default() -> Self {
        let (s,r) = crossbeam::channel::unbounded::<EvalRequest<G>>();
        thread::spawn(move ||{
            info!("Starting NullEvalServer");
            let n = NullEvaluator{};
            while let Ok(req) = r.recv(){
                let (p,v) = n.evaluate(&req.g);
                let resp = EvalResponse{
                    p:p,
                    v:v
                };
                req.resp_chan.send(resp).unwrap();
            }
            info!("NullEvalServer Shutting down");
        });
        Self{
            s_eval: s
        }        
    }
}


pub struct CTardEvalServer<G>{
    s_update: Sender<TardEvaluator>,
    s_eval:  Sender<EvalRequest<G>>
}

impl <G> CTardEvalServer<G>{
    pub fn update(&self, t: Tard){
        let eval = TardEvaluator::new(&t);
        self.s_update.send(eval).unwrap();
    }
}

impl <G:GameState> Server<G> for CTardEvalServer<G>
where
    G: 'static + Send + Clone
{
    type ClientType = CEvalClient<G>;

    fn get_client(&self) -> CEvalClient<G>{
        return CEvalClient{s_eval: self.s_eval.clone()};
    }
}

impl <G>Default for CTardEvalServer<G>
where
    G: GameState + Send
{
    fn default() -> Self{
        let (s_u, r_u) = crossbeam::channel::unbounded();
        let (s_e, r_e) = crossbeam::channel::unbounded();
        thread::spawn(move ||{
            start_tard_eval_server(r_u, r_e)
        });
        Self {
            s_update: s_u,
            s_eval: s_e
        }
    }
}

fn start_tard_eval_server<G:GameState>(
    r_u: Receiver<TardEvaluator>,
    r_e: Receiver<EvalRequest<G>>)
{
    info!("Starting TardEvalServer");
    debug!("Awaiting Tard Update");
    let t = r_u.recv().unwrap();
    debug!("Got Tard Update");
    while let Ok(req) = r_e.recv(){
        let (p,v) = t.evaluate(&req.g);
        let resp = EvalResponse{
            p:p,
            v:v
        };
        req.resp_chan.send(resp).unwrap();
    }
    info!("TardEvalServer shutting down")    
}


pub struct CBatchedTardEvalServer<G>{
    s_update: Sender<Tard>,
    s_eval:  Sender<EvalRequest<G>>
}


impl <G> CBatchedTardEvalServer<G>{
    pub fn update(&self, t: &Tard){
        debug!("Sending BTE Server updated tard");
        match self.s_update.send(t.valid()){
            Ok(_) => {}
            Err(_) => {
                warn!("Send Error when update BTE Server Tard");
            }
        }
    }
}


impl <G:GameState> Server<G> for CBatchedTardEvalServer<G>{
    type ClientType = CEvalClient<G>;
    
    fn get_client(&self) -> CEvalClient<G>{
        return CEvalClient{s_eval: self.s_eval.clone()};
    }
}


impl <G>Default for CBatchedTardEvalServer<G>
where
    G: GameState + Send
{
    fn default() -> Self{
        let (s_u, r_u) = crossbeam::channel::unbounded();
        let (s_e, r_e) = crossbeam::channel::unbounded();
        thread::spawn(move ||{
            start_batched_tard_eval_server(r_u, r_e)
        });
        Self {
            s_update: s_u,
            s_eval: s_e
        }
    }
}


fn start_batched_tard_eval_server<G:GameState>(
    r_u: Receiver<Tard>,
    r_e: Receiver<EvalRequest<G>>)
{
    thread:: spawn(move ||{
        info!("BTE reciever thread starting");
        info!("BTE awaiting Tard");
        let mut is_running = true;
        let mut t = r_u.recv().unwrap();
        info!("BTE recieved Tard");
        let device = t.devices()
                      .into_iter()
                      .next()
                      .unwrap();
        const BATCH_SIZE: usize = 512;
        let mut q = VecDeque::<EvalRequest<G>>::new();
        while is_running{    
            // Check for Tard updates
            trace!("BTE Recieve checking for Tard Updates");
            let (t_updated, recv_connected) = check_for_model_update(r_u.clone(), t);
            t = t_updated;
            is_running &= recv_connected;

            // Perform batch requests
            let (n, recv_connected) = read_queue(&r_e, &mut q);
            trace!("Added {} elements to BTE queue", n);
            is_running &= recv_connected;
            while q.len() > 0{
                let batch_size = std::cmp::min(q.len(), BATCH_SIZE);
                trace!{"Computing BTE batch of size {}", batch_size};
                let b_tens = batch_eval_tensor(&q, &device, batch_size);
                let (p_batch, v_batch) = t.forward(b_tens);
                let mut p_batch_vec = p_batch.exp()
                                            .into_data()
                                            .try_into_vec::<f32>()
                                            .unwrap(); 
                let mut v_batch_vec = v_batch.into_data()
                                            .try_into_vec::<f32>()
                                            .unwrap();
                // count backward because efficient to delete from end of vector          
                for i in (0..batch_size).rev(){
                    let p = p_batch_vec.split_off(p_batch_vec.len() - G::ACTN_SPC_SIZE);
                    let p = legalize_policy(p, &q[i].g);
                    let v = v_batch_vec.pop().unwrap();
                    trace!("Batch element {}/{} has p: {:?}, v: {}",
                            i+1, batch_size, &p, &v);
                    let resp = EvalResponse{
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
                q.drain(0..batch_size);
            }
        }
        debug!("BatchedTardEvaluator server shutting down");
    });
}

fn check_for_model_update(r_tard: Receiver<Tard>, t_in: Tard) -> (Tard, bool){
    let mut is_connected = true;
    let mut t = t_in;
    loop{
        match r_tard.try_recv(){
            Ok(t_updated) => {
                info!("BTE updating model");
                t = t_updated;
            }
            Err(TryRecvError::Empty) => {
                break; // nothing to update
            }
            Err(TryRecvError::Disconnected) => {
                info!("BTE tard receive channel closed ... signaling terminate for BTE thread");
                is_connected = false;
                break;
            }
        }
    }
    return (t, is_connected);
}

fn batch_eval_tensor <G:GameState>(
    q: &VecDeque<EvalRequest<G>>,
    device: &Device,
    batch_size: usize
) -> Tensor<4>{
    let mut v = Vec::with_capacity(batch_size * G::N_CHAN * G::ROWS * G::COLS);
    //let mut b_vec: Vec<Tensor<3>> = Vec::with_capacity(batch_size);
    for i in 0..batch_size{
        let req = &q[i];
        v.extend_from_slice(&req.g.to_tensor());
    }
    let td = TensorData::new(v, [batch_size, G::N_CHAN, G::ROWS, G::COLS]);
    let t = Tensor::<4>::from_data(td, device);
    return t;
}


// Receives all elements currently in the channel and adds them to queue
// then returns. If the queue is currently empty, it will block until
// at least 1 element is read. Otherwise non-blocking
// 
// Returns 
//      Number of elements read into queue.
//      is_connected
fn read_queue<G:GameState>(
    r: &Receiver<EvalRequest<G>>,
    q: &mut VecDeque<EvalRequest<G>>) -> (usize, bool)
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
            Err(RecvError) => { // disconnected
                return (0, false);
            }
        }
    }
    loop{
        match r.try_recv(){
            Ok(req) => {
                q.push_back(req)
            }
            Err(TryRecvError::Empty) => {
                break;
            }
            Err(TryRecvError::Disconnected) => {
                is_connected = false;
                break;
            }
        }
    }
    return (q.len() - n, is_connected);
}


#[derive(Debug)]
struct EvalResponse{
    p: Vec<f32>,
    v: f32
}

#[derive(Debug)]
struct EvalRequest<G>{
    g: G,
    resp_chan: Sender<EvalResponse>    // one-shot response channel
}


////////////////////////////////////////////////////////////////////
//
//                      Decentralized Evaluator
//
////////////////////////////////////////////////////////////////////
// This architecture features individual (decentralized) evaluator servers
// each capable of performing evaluations. These evaluators receive updates
// from the "decentralized server".


#[derive(Clone)]
struct DEvalClient<E>{
    eval: Option<E>,
    r_update: Receiver<E>
}


impl <E, G> Evaluator<G> for DEvalClient<E>
where 
    G: GameState,
    E: Evaluator<G>,
{
    fn evaluate (&self, g: &G) -> (Vec<f32>, f32) {
        match &self.eval {
            Some(e) => {
                e.evaluate(g)
            }
            None => { 
                NullEvaluator{}.evaluate(g)
            }
        }
    }
}


struct DTardServer <E, G>{
    s_update: Sender<E>,    // input
    s_cli: Sender<ReceiverRequest<E>>,
    
    _marker: PhantomData<G>,
}


impl <E, G> DTardServer<E,G>
where 
    E: 'static + Send + Clone
{
    pub fn new(&self) -> Self{
        let (s_u, r_u) = crossbeam::channel::unbounded();
        let (s_cli, r_cli) = crossbeam::channel::unbounded();
        start_dtard_server(r_u, r_cli);
        Self {
            s_update: s_u,
            s_cli: s_cli,
            _marker: PhantomData::<G>,
        }
        
    }
}

impl <E, G> Server<G> for DTardServer<E, G>
where 
    G:GameState,
    E:Evaluator<G>
{
    type ClientType = DEvalClient<E>;
    fn get_client(&self) -> Self::ClientType {
        let (s_resp, r_resp) = crossbeam::channel::bounded(1);
        let r_req = ReceiverRequest{
            resp_chan: s_resp,
        };
        self.s_cli.send(r_req).unwrap();
        let r_cli = r_resp.recv().unwrap();
        let cli = DEvalClient{
            eval:None,
            r_update: r_cli,
        };
        return cli;
    }
}

// FML I hate this. TODO maybe come back and fix later lol.
fn start_dtard_server<E>(
    r_u: Receiver<E>,
    r_cli: Receiver<ReceiverRequest<E>>)
where 
    E: 'static + Send + Clone
{
    thread::spawn(move||{ 
        let mut v: Vec<Sender<E>> = Vec::new();
        loop{
            crossbeam::channel::select! { // Perform blocking read on two channels at once
                // broadcast update to all clients
                recv(r_u) -> msg =>{
                    let e = msg.unwrap();
                    for s in v.iter(){
                        s.send(e.clone()).unwrap();
                    }
                }
                
                // Create new client channel. Keep sender, return reciever
                recv(r_cli)-> msg =>{
                    let req = msg.unwrap();
                    let (s,r) = crossbeam::channel::unbounded();
                    req.resp_chan.send(r).unwrap();
                    v.push(s);
                }
            }
        }
    });
}

struct ReceiverRequest<E>{
    resp_chan: Sender<Receiver<E>>
}



