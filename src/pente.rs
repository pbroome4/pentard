use std::{fmt};
use tracing::{trace, debug, info};
use rand::prelude::*;
use tch::{Kind::{self, Float}, Tensor};

use crate::engine::GameState;

pub const ROWS:usize = 9;
pub const COLS:usize = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece{
    Empty = 0,
    Black = 1,
    White = 2,
    Sentinel = 3, 
}

#[derive(Clone)]
pub struct PenteState{
    board_: [[Piece; COLS]; ROWS],
    b_caps_: u8, 
    w_caps_: u8,
    is_black_: bool,
    num_pieces_: usize, // Number of pieces actively on the board.
                        // Is redundant, since we could loop over board,
                        // but maintined to minimize computation
    is_terminal_: Piece, // Empty is non-terminal.
                         // Colored Piece is terminal winners color.
                         // Sentinel is draw.
}

impl GameState for PenteState{
    const ACTN_SPC_SIZE: usize = ROWS*COLS;
    const N_TENSOR_CHANNELS: usize = 4;

    fn new() -> Self{
        return PenteState::new();
    }

    // Output (num_channel=4) x ROWS x COLS Tensor.
    // channel 0 is occupancy mask of active player
    // channel 1 is occupancy mask of inactive player
    // channel 2 is captures of active player
    // channel 3 is captures of inactive player
    fn to_tensor(&self) -> Tensor {
        let mut a = [[[0.0; COLS]; ROWS]; Self::N_TENSOR_CHANNELS];
        let b_chan = if self.is_black() {0} else {1};
        let w_chan = if self.is_black() {1} else {0}; 
        for r in 0..ROWS{
            for c in 0..COLS{
                if self.board_[r][c] == Piece::Black{
                    a[b_chan][r][c] = 1.0;
                }else if self.board_[r][c] == Piece::White{
                    a[w_chan][r][c] = 1.0;
                }
                a[b_chan+2][r][c] = self.b_caps_ as f32;
                a[w_chan+2][r][c] = self.w_caps_ as f32;
            }
        }
        let flat_slice = unsafe {
            std::slice::from_raw_parts(a.as_ptr() as *const f32, Self::N_TENSOR_CHANNELS * ROWS * COLS)
        };
        let t = Tensor::from_slice(flat_slice).reshape(&[4 as i64, ROWS as i64, COLS as i64]);
        return t;
    }

    fn legal_actions_mask(&self) -> tch::Tensor {
        let mut v = [0; Self::ACTN_SPC_SIZE];
        for i in 0..Self::ACTN_SPC_SIZE{
            if self.is_legal_action(i){
                v[i] = 1;
            }
        }
        let t = Tensor::from_slice(&v);
        return t;
    }

    // Must only call in non-terminal state. 
    // Calling in a terminal state is UB
    // (here we opt to return usize::MAX so errors are visible quickly)
    fn get_random_action(&self) -> usize{
        let mut action = usize::MAX;
        let mut actions_count = 0;
        for row in 0..ROWS{
            for col in 0..COLS{
                if self.board_[row][col] == Piece::Empty{
                    actions_count += 1;
                    if rand::rng().random_range(0..actions_count) == 0{
                        action = coord_2_action(row,col);
                    }
                }
            }
        }
        return action;
    }
    
    fn apply_action(&mut self, action: usize){
        trace!("Applying action {}", action);
        debug_assert!(self.is_legal_action(action), "Illegal action {} for board \n{}", action, self);
        let (row,col) = action_2_coord(action);
        if self.is_black_{
            self.board_[row][col] = Piece::Black;
        }else{
            self.board_[row][col] = Piece::White;
        }
        self.num_pieces_ += 1;
        self.update_captures(row, col);
        if self.b_caps_ >= 5 || self.w_caps_ >= 5 || self.is_con5(row,col){
            self.is_terminal_ = self.get_active_piece()
        }else if self.num_pieces_ == ROWS*COLS{
            self.is_terminal_ = Piece::Sentinel;
        }
        self.is_black_ = !self.is_black_;
    }
    
    fn is_terminal(&self) -> bool{
        return self.is_terminal_ != Piece::Empty;
    }
    
    
    fn get_terminal_value(&self) -> i32 {
        // If it is your turn in terminal state, its a draw or a loss.
        if self.is_terminal_ == Piece::Sentinel{
            return 0;
        }
        return -1; 
    }


    fn get_objective_terminal_value(&self) -> i32{
        if self.is_terminal_ == Piece::Black{
            return 1;
        }else if self.is_terminal_ == Piece::White{
            return -1;
        }
        return 0;
    }


}


impl PenteState{
    pub fn new() -> Self{
        trace!("Pente new()");
        Self{
            board_: [[Piece::Empty; COLS]; ROWS],
            b_caps_:  0,
            w_caps_:  0,
            is_black_: true,
            num_pieces_: 0,
            is_terminal_: Piece::Empty,
        }
    }

    // Current player plays at provided space
    pub fn play(&mut self, row: usize, col: usize) -> bool{
        let action = coord_2_action(row,col);
        if self.is_legal_action(action){
            self.apply_action(action);
            return true;
        }
        return false;
    }

    pub fn is_legal_action(&self, action: usize) -> bool{
        let (row, col) = action_2_coord(action);
        return row < ROWS && col < COLS && self.board_[row][col] == Piece::Empty;
    }
    
    fn legal_actions(&self) -> Vec<usize>{
        let mut v = Vec::with_capacity(ROWS*COLS);
        for row in 0..ROWS{
            for col in 0..COLS{
                if self.board_[row][col] == Piece::Empty{
                    v.push(coord_2_action(row,col));
                }
            }
        }
        return v;
    }

    pub fn active_plyr_str(&self) -> &str{
        const PLYR_MAP: [&str; 2] = ["White", "Black"];
        return PLYR_MAP[self.is_black_ as usize];
    }

    // Is current player black
    pub fn is_black(&self) -> bool{
        return self.is_black_;
    }

    // If current player played at row,col, would it be connect-5
    fn is_con5(&self, row: usize, col: usize) -> bool{
        let active = self.get_active_piece();
        let mut count:u32 = 1; // Vertical
        let r = row as i32;
        let c = col as i32;
        const DIRS: [(i32, i32); 4] = [
            (1, 0), (0, 1),   // Vert/Horiz
            (1, 1), (1, -1),  // Diagonals
        ];
        
        for (dr, dc) in DIRS.iter(){
            count = 1;
            for i in 1..5{
                if self.get_square(r + dr * i, c + dc * i) == Some(active){
                    count+=1;
                } else{ break;}
            }
            for i in 1..5{
                if self.get_square(r - dr * i, c - dc * i) == Some(active){
                    count+=1;
                } else{ break;}
            }
            if count >= 5{
                return true;
            }
        }
        return false;
    }

    // Safe method to grab piece occupying space. 
    // If space is out of bounds it returns None.
    fn get_square(&self, row: i32, col: i32) -> Option<Piece>{
        if row < 0 || row >= (ROWS as i32) || col < 0 || col >= (COLS as i32) {
            return None
        }
        let r = row as usize;
        let c = col as usize;
        return Some(self.board_[r][c]);
    }

    // Check for captures due to current player playing at provided space,
    // and remove the captured pieces.
    fn update_captures(&mut self, row: usize, col: usize) {
        let active = self.get_active_piece();
        let inactive = self.get_inactive_piece();
        let r = row as i32;
        let c = col as i32;
        const DIRS: [(i32, i32); 8] = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   // Vert/Horiz
            (-1, -1), (1, 1), (-1, 1), (1, -1)  // Diagonals
        ];
        for (dr, dc) in DIRS.iter() {
            if self.get_square(r + dr, c + dc) == Some(inactive) &&
            self.get_square(r + dr * 2, c + dc * 2) == Some(inactive) &&
            self.get_square(r + dr * 3, c + dc * 3) == Some(active) 
            {
                self.board_[(r + dr) as usize][(c + dc) as usize] = Piece::Empty;
                self.board_[(r + dr * 2) as usize][(c + dc * 2) as usize] = Piece::Empty;
                self.add_captures(self.is_black_, 1);
                self.num_pieces_ -= 2;
            }
        }
    }
    
    // Change players captures by delta.
    fn add_captures(&mut self, is_black:bool, delta: u8){
        if is_black{
            self.b_caps_ += delta;
        }else{
            self.w_caps_ += delta;
        }
    }

    // Get active player as a Piece
    fn get_active_piece(&self) -> Piece{
        const PIECE_MAP: [Piece; 2] = [Piece::White, Piece::Black];
        return PIECE_MAP[self.is_black_ as usize]
    }

    // Get inactive player as a Piece
    fn get_inactive_piece(&self) -> Piece{
        const PIECE_MAP: [Piece; 2] = [Piece::Black, Piece::White];
        return PIECE_MAP[self.is_black_ as usize]
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
        writeln!(f, "Black captures:{}   White Captures:{}", self.b_caps_, self.w_caps_)?;
        Ok(())
    }
}


