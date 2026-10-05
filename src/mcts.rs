#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(pub u32);


pub struct MctsNode<A> {
    pub action: Option<A>,            // The move that led to this state
    pub parent: Option<NodeId>,       // Parent index for backpropagation
    pub first_child: Option<NodeId>,  // First element in the sibling list
    pub next_sibling: Option<NodeId>, // Next sibling index
    pub visits: u32,                  // Atomic or scalar counters
    pub total_value: f32,             // Accumulated evaluation value
}

pub struct MctsTree<A> {
    pub nodes: Vec<MctsNode<A>>,   // The Memory Pool Arena
}