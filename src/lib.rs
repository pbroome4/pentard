use std::fmt;


pub const ROWS:usize = 9;
pub const COLS:usize = 9;





#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece{
    Empty = 0,
    Black = 1,
    White = 2,
}

pub trait GameState{
    // Get vector of legal actions 
    fn get_actions(&self) -> Vec<usize>;

    // Apply an action. action MUST be a legal action.
    fn apply_action(&self, action: usize);

    // Is the game over
    fn is_terminal(&self) -> bool;

    // Get the terminal value
    fn get_terminal_value(&self) -> i32;
 
    // Perform a deep copy of the GameState
    fn clone(&self) -> Self;     
}

pub struct PenteState{
    board_: [[Piece; COLS]; ROWS],
    b_caps_: u8, 
    w_caps_: u8,
    is_black_: bool,
    num_pieces_: u8, 
}

impl PenteState{
    pub fn new() -> Self{
        Self{
            board_: [[Piece::Empty; COLS]; ROWS],
            b_caps_:  0,
            w_caps_:  0,
            is_black_: true,
            num_pieces_: 0,
        }
    }


    fn coord_2_action(row: usize, col:usize) -> usize{
        row*COLS + col
    }


    fn action_2_coord(action: usize) -> (usize, usize){
        let row = action/COLS;
        let col = action - row*COLS;
        (row, col)
    }


    pub fn play(&mut self, row: usize, col: usize){
        let action = PenteState::coord_2_action(row, col);
        if self.get_actions().contains(&action){
            if self.is_black_{
                self.board_[row][col] = Piece::Black;
            }else{
                self.board_[row][col] = Piece::White;
            }
            self.num_pieces_ += 1;
            self.is_black_ = !self.is_black_;
        }
        //action = PenteState.coord_2_action(row, col);
        //self.apply_action(action);
    }
}

impl PenteState{
    fn get_actions(&self) -> Vec<usize>{
        let mut v = Vec::with_capacity(ROWS*COLS);
        for row in 0..ROWS{
            for col in 0..COLS{
                if self.board_[row][col] == Piece::Empty{
                    v.push(PenteState::coord_2_action(row,col));
                }
            }
        }
        return v;
    }
    //fn apply_action(&self, action: usize);
    //fn is_terminal(&self) -> bool;
    //fn get_terminal_value(&self) -> isize;
    //fn clone(&self) -> Self; 
}



impl fmt::Display for PenteState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..ROWS{
            for col in 0..COLS{
                if self.board_[row][col] == Piece::Empty{
                    write!(f, "· ")?;
                }else if self.board_[row][col]  == Piece::Black{
                    write!(f, "B ")?;
                }else{
                    write!(f, "W ")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}



//use pyo3::prelude::*;
//
///// A Python module implemented in Rust.
//#[pymodule]
//mod engine {
//    use pyo3::prelude::*;
//
//    /// Formats the sum of two numbers as string.
//    #[pyfunction]
//    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//        Ok((a + b).to_string())
//    }
//}
