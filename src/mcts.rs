use rand;
use crate::engine;

type NodeId = usize;


pub struct MctsNode<A> {
    pub id: NodeId,
    pub action_: Option<A>,            // The move that led to this state
    pub parent_: Option<NodeId>,       // Parent index for backpropagation
    pub first_child_: Option<NodeId>,  // First element in the sibling list
    pub next_sibling_: Option<NodeId>, // Next sibling index
    pub visits_: u32,                  // Atomic or scalar counters
    pub total_value_: f32,             // Accumulated evaluation value
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
            id: 0,
            action_:  None,
            parent_: None,
            first_child_: None,     
            next_sibling_: None,
            visits_: 0,
            total_value_: 0.0,
        };
        x.nodes_.push(root);
        return x;
    }

    pub fn get(&self, id: NodeId) -> Option<& MctsNode<A>>{
        return self.nodes_.get(id);
    }

    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut MctsNode<A>>{
        return self.nodes_.get_mut(id);
    }

    pub fn is_expanded(&self, node_id: NodeId) -> Option<bool>{
        if let Some(node) = self.get(node_id){
            return Some(node.first_child_ == None);
        }
        return None;
    }

    pub fn expand(&mut self, node_id: NodeId, actions: Vec<A>){
        for action in actions.into_iter(){{
                self.add_child(action, node_id);
            }
        }
    }


    pub fn pick_ucb(&self, node_id:NodeId) -> Option<NodeId>{
        // TODO
        // uses resevoir sampling to uniformly select argmax
        let node = self.get(node_id)?;
        let child_id = node.first_child_?;
        let mut ucb_id  = child_id;
        let mut ucb_val = self.calc_ucb(node_id, child_id);
        let mut count = 1;
        while let Some(child_id) = self.next_sibling(child_id){   
            let ucb_test = self.calc_ucb(node_id, child_id);
            if ucb_test > ucb_val{
                count = 0;
                ucb_val = ucb_test;
                ucb_id = child_id;
            }else if ucb_test == ucb_val{
                count += 1;
                let p= 1.0 / (count as f32);
                if rand::random::<f32>() < p{    // resevoir sampling
                    ucb_id = child_id;
                }
            }
        }
        return None;
    }

    fn calc_ucb(&self, parent: NodeId, child: NodeId) -> Option<f32>{
        let C = (2.0_f32).sqrt();
        let pnode = self.get(parent)?;
        let cnode = self.get(child)?;
        let mut ret = Some(f32::INFINITY);
        if cnode.visits_ > 0{
            ret = Some((cnode.total_value_ / (cnode.visits_ as f32)) +
                   C * ((pnode.visits_ / cnode.visits_) as f32).ln().sqrt() );
        }
        return ret;
    }

    fn next_sibling(&self, node_id:NodeId) -> Option<NodeId>{
        let node = self.get(node_id)?;
        return node.next_sibling_;
    }

    // Create a new node in the arena-pool. Return it's id.
    fn add_child(&mut self, action: A, parent: NodeId) -> Option<&mut MctsNode<A>>{
        let child_id: NodeId = self.nodes_.len();
        let child = MctsNode{
            id: child_id,
            action_: Some(action),
            parent_: Some(parent),
            first_child_: None,
            next_sibling_: self.nodes_[parent].first_child_,
            visits_: 0,
            total_value_: 0.0,
        };
        self.nodes_.push(child);
        self.nodes_[parent].first_child_ = Some(child_id);
        return self.get_mut(child_id);
    }
}


fn mcts_sim<A>(tree: &mut MctsTree<A>, state: &impl engine::GameState<A>) -> Option<bool>{
    let mut state_copy = state.clone();
    let node_id = 0; // Root
    while !state.is_terminal(){
        let node = tree.get(node_id)?;
        let is_expanded = tree.is_expanded(node_id)?;
        if node.visits_ >= 1 && !is_expanded {
            tree.expand(node_id, state.get_actions());
        }
        let node = tree.get(node_id)?;
        if node.visits_ > 0{
            //pick ucb
            //continue;
        }
        //random rollout
    }
    return Some(true);
}


fn mcts<A>(state: &impl engine::GameState<A>, num_sims: u64){
    let mut tree: MctsTree<A> = MctsTree::new();
    for i in 0..num_sims{
        //mcts_sim(tree, state);
        print!("Simulation {}", i);
    }

}