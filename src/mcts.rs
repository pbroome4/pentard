use rand::prelude::*;
use crate::engine;
use std::fmt;

type NodeId = usize;


pub struct MctsNode<A> {
    pub id_: NodeId,
    pub action_: Option<A>,            // The move that led to this state
    pub parent_: Option<NodeId>,       // Parent index for backpropagation
    pub first_child_: Option<NodeId>,  // First element in the sibling list
    pub next_sibling_: Option<NodeId>, // Next sibling index
    pub visits_: u32,                  // Atomic or scalar counters
    pub total_value_: f32,             // Accumulated evaluation value
    pub is_expanded_: bool,
}

impl<A> fmt::Display for MctsNode<A>{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut avg_val = 0.0;
        if self.visits_ > 0{
            avg_val = self.total_value_ / (self.visits_ as f32);
        }
        writeln!(f, "node_id: {}   visits: {}   avg_value: {}   is_expanded: {}",self.id_, self.visits_, avg_val, self.is_expanded_);
        return Ok(());
    }
}


pub struct MctsTree<A> {
    pub nodes_: Vec<MctsNode<A>>,   // The Memory Pool Arena
}


impl<A> MctsTree<A>{
    // An Arena-Pool implementation of a MCTS Tree.

    // Create the Arena-Pool. Add the root Node with id==0.
    pub fn new() -> Self{
        let mut x = Self{
            nodes_: Vec::new(),
        };
        let root = MctsNode{
            id_: 0,
            action_:  None,
            parent_: None,
            first_child_: None,     
            next_sibling_: None,
            visits_: 0,
            total_value_: 0.0,
            is_expanded_: false,
        };
        x.nodes_.push(root);
        return x;
    }

    pub fn get(&self, node_id:NodeId) -> Option<&MctsNode<A>>{
        return self.nodes_.get(node_id);
    }

    pub fn get_mut(&mut self, node_id:NodeId) -> Option<&mut MctsNode<A>>{
        return self.nodes_.get_mut(node_id);
    }

    pub fn is_expanded(&self, node_id: NodeId) -> Option<bool>{
        let node = self.nodes_.get(node_id)?;
        return Some(node.first_child_ == None);
    }

    pub fn expand(&mut self, node_id: NodeId, actions: Vec<A>){
        for action in actions.into_iter(){{
                self.add_child(action, node_id);
            }
        }
        if let Some(node) = self.nodes_.get_mut(node_id){
            node.is_expanded_ = true;
        }
    }

    pub fn pick_ucb(&self, node_id:NodeId) -> Option<NodeId>{
        // uses resevoir sampling to uniformly select argmax
        println!("Picking ucb move");
        let mut child_id = self.get(node_id)?.first_child_?;
        let mut ucb_id  = child_id;
        let mut ucb_val = self.calc_ucb(node_id, child_id)?;
        let mut count = 1;
        loop{
            let Some(next_child_id) = self.get(child_id)?.next_sibling_ else { break;};
            child_id = next_child_id;
            let ucb_test = self.calc_ucb(node_id, child_id)?;
            println!("ucb_id: {}  ucb_val: {}   ucb_test id: {}  val:{}",ucb_id, ucb_val, child_id, ucb_test);
            if ucb_test > ucb_val{
                count = 1;
                ucb_val = ucb_test;
                ucb_id = child_id;
            }else if ucb_test == ucb_val{
                count += 1;
                let p = 1.0 / (count as f32);
                let mut rng = rand::rng();
                if rng.random_range(0.0..1.0) < p {    // resevoir sampling
                    ucb_id = child_id;
                }
            }
        }
        return Some(ucb_id);
    }

    pub fn backprop(&mut self, leaf_id: NodeId, reward:i32) -> Option<()>{
        let mut node_id = leaf_id;
        loop{
            let node = self.get_mut(node_id)?;
            node.total_value_ += reward as f32;
            node.visits_ += 1;
            println!("{}", node);
            let Some(parent_id) = node.parent_ else {break};
            node_id = parent_id;
        }
        return Some(());
    }

    fn calc_ucb(&self, parent: NodeId, child: NodeId) -> Option<f32>{
        let c = (2.0f32).sqrt();
        let pnode = self.nodes_.get(parent)?;
        let cnode = self.nodes_.get(child)?;
        let mut ret = Some(f32::INFINITY);
        if cnode.visits_ > 0{
            ret = Some((cnode.total_value_ / (cnode.visits_ as f32)) +
                   c * ((pnode.visits_ / cnode.visits_) as f32).ln().sqrt() );
        }
        return ret;
    }


    // Create a new node in the arena-pool. Return it's id.
    fn add_child(&mut self, action: A, parent: NodeId) -> NodeId{
        let child_id: NodeId = self.nodes_.len();
        let child = MctsNode{
            id_: child_id,
            action_: Some(action),
            parent_: Some(parent),
            first_child_: None,
            next_sibling_: self.nodes_[parent].first_child_,
            visits_: 0,
            total_value_: 0.0,
            is_expanded_: false,
        };
        self.nodes_.push(child);
        self.nodes_[parent].first_child_ = Some(child_id);
        println!("Added child {}", child_id);
        return child_id;
    }
}


fn random_rollout<A>(state: &mut impl engine::GameState<A>) -> i32{
    while !state.is_terminal(){
        let actions = state.get_actions();
        let mut rng = rand::rng();
        let rand_index = rng.random_range(0..actions.len()); 
        let action = &actions[rand_index];
        state.apply_action(&action);
    }
    return state.get_terminal_value();
}

fn mcts_sim<A>(tree: &mut MctsTree<A>, state_orig: &impl engine::GameState<A>, max_depth: usize) -> Option<()>{
    let mut state = state_orig.clone();
    let mut node_id = 0; // Root
    let mut depth = 0;
    loop{ // Drill down to a leaf node.
        if depth >= max_depth {break;};
        if let Some(node) = tree.get(node_id){
            if node.visits_>= 1 && !node.is_expanded_{
                println!("Expanding {}", node);
                tree.expand(node_id, state.get_actions());
                let node = tree.get(node_id)?;
                println!("Expanded {}", node);
            }
        }
        let node = tree.get(node_id)?;
        if !node.is_expanded_{ break;}
        let Some(next_node_id) = tree.pick_ucb(node_id) else {break;};
        println!("ucb id: {}", next_node_id);
        let next_node = tree.get(next_node_id)?;
        let Some(action) = next_node.action_.as_ref() else { break;};
        state.apply_action(action);
        node_id = next_node_id;
        depth +=1;
        println!("depth {}", depth);
    } // node_id should be leaf node.
    let reward = random_rollout(&mut state);
    println!("reward: {}", reward);
    tree.backprop(node_id, reward);    
    return Some(());
}


pub fn mcts<A>(state: &impl engine::GameState<A>, num_sims: u64) -> Option<()>{
    let mut tree: MctsTree<A> = MctsTree::new();
    for i in 0..num_sims{
        mcts_sim(&mut tree, state, 100);
    }
    let root = tree.get(0)?;
    println!("{}", root);
    return Some(());
}