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


pub struct GameState{
    board_: [[Piece; COLS]; ROWS],
    b_caps_: u8, 
    w_caps_: u8,
    is_black_: bool,
    num_pieces_: u8, 
}

impl GameState{
    pub fn new() -> Self{
        Self{
            board_: [[Piece::Empty; COLS]; ROWS],
            b_caps_:  0,
            w_caps_:  0,
            is_black_: true,
            num_pieces_: 0,
        }
    }
}

impl fmt::Display for GameState {
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
