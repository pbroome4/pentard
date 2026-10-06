use crate::engine;

type NodeId = usize;


pub struct MctsNode<A> {
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

impl<A> MctsNode<A>{
}

impl<A> MctsTree<A>{
    // An Arena-Pool implementation of a MCTS Tree.

    // Create the Arena-Pool. Add the root Node with id==0.
    pub fn new() -> Self{
        let mut x = Self{
            nodes_: Vec::new(),
        };
        let root = MctsNode{
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

    
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut MctsNode<A>>{
        return self.nodes_.get_mut(0);
    }

    // Create a new node in the arena-pool. Return it's id.
    pub fn add_node(&mut self, action: A, parent: NodeId) -> NodeId{
        let child = MctsNode{
            action_: Some(action),
            parent_: Some(parent),
            first_child_: None,
            next_sibling_: self.nodes_[parent].first_child_,
            visits_: 0,
            total_value_: 0.0,
        };
        let child_id: NodeId = self.nodes_.len();
        self.nodes_.push(child);
        self.nodes_[parent].first_child_ = Some(child_id);
        return child_id;
    }
}

fn mcts_sim<A>(tree: &mut MctsTree<A>, action: A){
    
}

fn mcts(state: &impl engine::GameState, num_sims: u64){
    let mut tree: MctsTree<u32> = MctsTree::new();
    for i in 0..num_sims{
        //mcts_sim(tree, state);
        print!("Simulation {}", i);
    }

}