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
    num_pieces_: u8,  // Number of pieces activly on the board.
                      // Is redundant, since we could loop over board,
                      // but maintined to minimize computation
    is_terminal_: bool,
}

impl PenteState{
    pub fn new() -> Self{
        Self{
            board_: [[Piece::Empty; COLS]; ROWS],
            b_caps_:  0,
            w_caps_:  0,
            is_black_: true,
            num_pieces_: 0,
            is_terminal_: false,
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


    pub fn play(&mut self, row: usize, col: usize) -> bool{
        let action = PenteState::coord_2_action(row,col);
        if self.is_legal_action(action){
            if self.is_black_{
                self.board_[row][col] = Piece::Black;
            }else{
                self.board_[row][col] = Piece::White;
            }
            self.num_pieces_ += 1;
            self.update_captures(row, col);
            // check connect 5
            self.is_black_ = !self.is_black_;
            return true;
        }
        return false;
        //action = PenteState.coord_2_action(row, col);
        //self.apply_action(action);
    }

    pub fn is_legal_action(&self, action: usize) -> bool{
        return self.get_actions().contains(&action);
    }

    pub fn is_black(&self) -> bool{
        return self.is_black_;
    }

    
    fn update_captures(&mut self, row:usize, col:usize) {
        let active = self.get_active_piece();
        let inactive = self.get_inactive_piece();
        if row >= 3{ // vert up
            if self.board_[row-3][col] == active 
                    && self.board_[row-2][col] == inactive
                    && self.board_[row-1][col] == inactive{
                self.board_[row-2][col] = Piece::Empty;
                self.board_[row-1][col] = Piece::Empty;
                self.add_captures(self.is_black_, 1);
            }
        }
        if row < ROWS-3{ // vert down
            if self.board_[row+3][col] == active 
                    && self.board_[row+2][col] == inactive
                    && self.board_[row+1][col] == inactive{
                self.board_[row+2][col] = Piece::Empty;
                self.board_[row+1][col] = Piece::Empty;
                self.add_captures(self.is_black_, 1);
            }
        }
        if col >= 3{ // horiz left
            if self.board_[row][col-3] == active 
                    && self.board_[row][col-2] == inactive
                    && self.board_[row][col-1] == inactive{
                self.board_[row][col-2] = Piece::Empty;
                self.board_[row][col-1] = Piece::Empty;
                self.add_captures(self.is_black_, 1);
            }
        }
        if col < COLS - 3{ // horiz right
            if self.board_[row][col+3] == active 
                    && self.board_[row][col+2] == inactive
                    && self.board_[row][col+1] == inactive{
                self.board_[row][col+2] = Piece::Empty;
                self.board_[row][col+1] = Piece::Empty;
                self.add_captures(self.is_black_, 1);
            }
        }
        if row >= 3 && col >= 3{ // Diag top-left
            if self.board_[row-3][col-3] == active 
                    && self.board_[row-2][col-2] == inactive
                    && self.board_[row-1][col-1] == inactive{
                self.board_[row-2][col-2] = Piece::Empty;
                self.board_[row-1][col-1] = Piece::Empty;
                self.add_captures(self.is_black_, 1);
            }
        }
        if row >= 3 && col < COLS-3{ // Diag top-right
            if self.board_[row-3][col+3] == active 
                    && self.board_[row-2][col+2] == inactive
                    && self.board_[row-1][col+1] == inactive{
                self.board_[row-2][col+2] = Piece::Empty;
                self.board_[row-1][col+1] = Piece::Empty;
                self.add_captures(self.is_black_, 1);
            }
        }
        if row < COLS-3 && col < COLS-3{ // Diag bottom-right
            if self.board_[row+3][col+3] == active 
                    && self.board_[row+2][col+2] == inactive
                    && self.board_[row+1][col+1] == inactive{
                self.board_[row+2][col+2] = Piece::Empty;
                self.board_[row+1][col+1] = Piece::Empty;
                self.add_captures(self.is_black_, 1);
            }
        }
        if row < COLS-3 && col >= 3{ // Diag bottom-right
            if self.board_[row+3][col-3] == active 
                    && self.board_[row+2][col-2] == inactive
                    && self.board_[row+1][col-1] == inactive{
                self.board_[row+2][col-2] = Piece::Empty;
                self.board_[row+1][col-1] = Piece::Empty;
                self.add_captures(self.is_black_, 1);
            }
        }
    }
    
    fn add_captures(&mut self, is_black:bool, delta: u8){
        if is_black{
            self.b_caps_ += delta;
        }else{
            self.w_caps_ += delta;
        }
    }

    fn get_active_piece(&self) -> Piece{
        const PIECE_MAP: [Piece; 2] = [Piece::White, Piece::Black];
        return PIECE_MAP[self.is_black_ as usize]
    }


    fn get_inactive_piece(&self) -> Piece{
        const PIECE_MAP: [Piece; 2] = [Piece::Black, Piece::White];
        return PIECE_MAP[self.is_black_ as usize]
    }

    pub fn active_plyr_str(&self) -> &str{
        const PLYR_MAP: [&str; 2] = ["White", "Black"];
        return PLYR_MAP[self.is_black_ as usize];
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
    
    fn is_terminal(&self) -> bool{
        return self.is_terminal_;
    }
    //fn get_terminal_value(&self) -> isize;
    //fn clone(&self) -> Self; 
}



impl fmt::Display for PenteState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "   ")?;
        for col in 0..COLS{
            write!(f, "{:02} ", col)?;
        }
        write!(f,"\n")?;
        for row in 0..ROWS{
            write!(f, "{:02}  ", row)?;
            for col in 0..COLS{
                if self.board_[row][col] == Piece::Empty{
                    write!(f, "·  ")?;
                }else if self.board_[row][col]  == Piece::Black{
                    write!(f, "B  ")?;
                }else{
                    write!(f, "W  ")?;
                }
            }
            writeln!(f)?;
            }
        writeln!(f, "Black captures:{}   White Captures{}", self.b_caps_, self.w_caps_)?;
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
