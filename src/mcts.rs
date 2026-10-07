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
    pub is_expanded: bool,
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
            is_expanded: false,
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
            node.is_expanded = true;
        }
    }

    pub fn pick_ucb(&self, node_id:NodeId) -> Option<NodeId>{
        // uses resevoir sampling to uniformly select argmax
        let child_id = self.get(node_id)?.first_child_?;
        let mut ucb_id  = child_id;
        let mut ucb_val = self.calc_ucb(node_id, child_id);
        let mut count = 1;
        while let Some(child_id) = self.get(child_id)?.next_sibling_{   
            let ucb_test = self.calc_ucb(node_id, child_id);
            if ucb_test > ucb_val{
                count = 1;
                ucb_val = ucb_test;
                ucb_id = child_id;
            }else if ucb_test == ucb_val{
                count += 1;
                let p = 1.0 / (count as f32);
                if rand::random::<f32>() < p {    // resevoir sampling
                    ucb_id = child_id;
                }
            }
        }
        return Some(ucb_id);
    }


    fn calc_ucb(&self, parent: NodeId, child: NodeId) -> Option<f32>{
        let C = (2.0_f32).sqrt();
        let pnode = self.nodes_.get(parent)?;
        let cnode = self.nodes_.get(child)?;
        let mut ret = Some(f32::INFINITY);
        if cnode.visits_ > 0{
            ret = Some((cnode.total_value_ / (cnode.visits_ as f32)) +
                   C * ((pnode.visits_ / cnode.visits_) as f32).ln().sqrt() );
        }
        return ret;
    }


    // Create a new node in the arena-pool. Return it's id.
    fn add_child(&mut self, action: A, parent: NodeId) -> NodeId{
        let child_id: NodeId = self.nodes_.len();
        let child = MctsNode{
            id: child_id,
            action_: Some(action),
            parent_: Some(parent),
            first_child_: None,
            next_sibling_: self.nodes_[parent].first_child_,
            visits_: 0,
            total_value_: 0.0,
            is_expanded: false,
        };
        self.nodes_.push(child);
        self.nodes_[parent].first_child_ = Some(child_id);
        return child_id;
    }
}


fn random_rollout<A>(state: &mut impl engine::GameState<A>) -> i32{
    return 0;
}

fn mcts_sim<A>(tree: &mut MctsTree<A>, state_orig: &impl engine::GameState<A>, max_depth: usize) -> Option<()>{
    let mut state = state_orig.clone();
    let mut node_id = 0; // Root
    let mut depth = 0;
    loop{ // Drill down to a leaf node.
        if depth >= max_depth {break;};
        if let Some(node) = tree.get(node_id){
            if node.visits_>= 1 && !node.is_expanded{
                tree.expand(node_id, state.get_actions());
            }
        }
        let Some(node) = tree.get(node_id) else { return None;}; // Catostrophic
        if !node.is_expanded{ break;}
        let Some(next_node_id) = tree.pick_ucb(node_id) else {break;};
        let Some(next_node) = tree.get(next_node_id) else { return None;}; // catostrophic
        let Some(action) = next_node.action_.as_ref() else { break;};
        state.apply_action(action);
        node_id = next_node_id;
        depth +=1;
    } // node_id should be leaf node.
    let reward = random_rollout(&mut state);    
    return Some(());
}


pub fn mcts<A>(state: &impl engine::GameState<A>, num_sims: u64){
    //let mut tree: MctsTree<A> = MctsTree::new();
    //println!("Simulation start");
    //mcts_sim(&mut tree, &state, 100);
    //println!("Simulation over");
}