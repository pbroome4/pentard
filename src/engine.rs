use std::fmt;
use tracing::{trace, debug, info};


pub const ROWS:usize = 19;
pub const COLS:usize = 19;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece{
    Empty = 0,
    Black = 1,
    White = 2,
    Sentinel = 3, 
}

pub trait GameState<A>{
    // Get vector of legal actions 
    fn get_actions(&self) -> Vec<A>;

    // Apply an action. action MUST be a legal action.
    fn apply_action(&mut self, action: &A);

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
    num_pieces_: usize, // Number of pieces activly on the board.
                      // Is redundant, since we could loop over board,
                      // but maintined to minimize computation
    is_terminal_: Piece, // Empty is non-terminal.
                         // Colored Piece is terminal winners color.
                         // Sentinel is draw.
}

impl PenteState{
    pub fn new() -> Self{
        Self{
            board_: [[Piece::Empty; COLS]; ROWS],
            b_caps_:  0,
            w_caps_:  0,
            is_black_: true,
            num_pieces_: 0,
            is_terminal_: Piece::Empty,
        }
    }

    fn coord_2_action(row: usize, col:usize) -> usize{
        row*COLS + col
    }

    fn action_2_coord(action: &usize) -> (usize, usize){
        let row = action/COLS;
        let col = action - row*COLS;
        (row, col)
    }


    pub fn play(&mut self, row: usize, col: usize) -> bool{
        let action = PenteState::coord_2_action(row,col);
        if self.is_legal_action(&action){
            self.apply_action(&action);
            return true;
        }
        return false;
    }

    pub fn is_legal_action(&self, action: &usize) -> bool{
        let (row, col) = PenteState::action_2_coord(action);
        return row < ROWS && col < COLS && self.board_[row][col] == Piece::Empty;
    }

    pub fn active_plyr_str(&self) -> &str{
        const PLYR_MAP: [&str; 2] = ["White", "Black"];
        return PLYR_MAP[self.is_black_ as usize];
    }

    pub fn is_black(&self) -> bool{
        return self.is_black_;
    }

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


    fn get_square(&self, row: i32, col: i32) -> Option<Piece>{
        if row < 0 || row >= (ROWS as i32) || col < 0 || col >= (COLS as i32) {
            return None
        }
        let r = row as usize;
        let c = col as usize;
        return Some(self.board_[r][c]);
    }

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

    
}

impl GameState<usize> for PenteState{
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
    
    fn apply_action(&mut self, action: &usize){
        trace!("Applying action {}", action);
        let (row,col) = PenteState::action_2_coord(action);
        if self.is_black_{
            self.board_[row][col] = Piece::Black;
        }else{
            self.board_[row][col] = Piece::White;
        }
        self.num_pieces_ += 1;
        self.update_captures(row, col);
        if self.b_caps_ == 5 || self.w_caps_ == 5 || self.is_con5(row,col){
            self.is_terminal_ = self.get_active_piece()
        }else if self.num_pieces_ == ROWS*COLS{
            self.is_terminal_ = Piece::Sentinel;
        }
        self.is_black_ = !self.is_black_;
    }
    
    fn is_terminal(&self) -> bool{
        return self.is_terminal_ != Piece::Empty;
    }
    
    fn get_terminal_value(&self, ) -> i32{
        if self.is_terminal_ == Piece::Black{
            return 1;
        }else if self.is_terminal_ == Piece::White{
            return -1;
        }
        return 0;
    }


    fn clone(&self) -> Self{
        return Self{
            board_: self.board_.clone(),
            b_caps_: self.b_caps_,
            w_caps_: self.w_caps_,
            is_black_: self.is_black_,
            num_pieces_: self.num_pieces_,
            is_terminal_:self.is_terminal_
        }
    } 
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


