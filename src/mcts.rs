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

    pub fn expand_node(&mut self, node_id: NodeId, actions: Vec<A>){
        for action in actions.into_iter(){{
                self.add_child(action, node_id);
            }
        }
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
        if tree.is_expanded(node_id) == Some(false){
            tree.expand_node(node_id, state.get_actions());
        }
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