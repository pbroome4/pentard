from dataclasses import dataclass
import numpy as np
from enum import Enum


def fill(brd, chunk, chunk_rem):
    brd[chunk] |=  np.uint64(1) << chunk_rem


def _coord_2_chunk(row: np.uint8, col:np.uint8):
    action = _coord_2_action_id(row,col)
    return _action_id_2_chunk(action)


def _action_id_2_chunk(id):
    """
    id describes board location as index position on the vectorized board.

    Returns the chunk, chunk_rem.
        chunk refers to the index of the u64 bitboard containing the board square.
        chunk_rem is the bit index in this bit board for this board square.
    """
    occ_chunk = id // 64
    occ_chunk_rem = id - occ_chunk * 64
    return occ_chunk, occ_chunk_rem

def action_id_2_coord(id):    
    row = id // COLS
    col = id - row*COLS
    return row, col

def _coord_2_action_id(row, col):
    return row*COLS + col

def _chunk_2_coord(chunk, chunk_rem):
    bit_pos = chunk * 64 + chunk_rem
    row = bit_pos / COLS
    col = bit_pos - row*COLS
    return (row,col)


def _coord_inbounds(row, col):
    return row >= 0 and row < ROWS and col >= 0 and col < COLS



def init_brd_mask():
    """ Return a bitboard masking valid squares on the board. 0 for invalid squares. 1 for valid squares """
    brd = np.zeros((BRD_LEN,), dtype=np.uint64)
    for row in range(ROWS):
        for col in range(COLS):
            chunk, chunk_rem = _coord_2_chunk(row, col)
            fill(brd, chunk, chunk_rem)
    return brd


# Board is a 19 x 19. We will store this as a bitboard using u64 ints.
# This requires 7 u64's to store occupancies for each player. 
ROWS = 9
COLS = 9       
BRD_LEN = int(np.ceil(ROWS*COLS/64))
BRD_MASK = init_brd_mask()


class State(Enum):
    Running = 0
    White = 1
    Black = 2
    Draw = 3

   
class GameState:
    """
    User Guide:
    Defines a lean Pente Board State for an Alphazero style Interface.
    We implement the board using bitboards to minimize memory footprint.
    The Action space |A| is set(list) of ROWS*COLS elements, one for each location.

      
    Developer Guide:
    Action ID's start at 0 in the top left corner and count up across columns and then across the rows.
    The board is chunked down into 64 bit occupancy bit masks to minimize memory footprint and hopefully improve performance.
    If this proves fruitful, this should probably get ported to a faster language like c/c++/rust
    """
    def __init__(self):
        self.w_occs_ = np.zeros((BRD_LEN,), dtype=np.uint64)
        self.w_caps_ = np.uint8(0)
        self.b_occs_ = np.zeros((BRD_LEN, ), dtype=np.uint64)
        self.b_caps_ = np.uint8(0)
        self.num_pieces = 0 # num pieces currently on the board
        self.is_white_ = True # True for whites move, False for blacks move.
        self.game_state_ = State.Running 

    def to_tensor(self):
        """
        Returns a tensor t with dimension 4 x ROWS x COLS representing the board state for training.
        This operation is lossy.
        T[1] is a ROWS x COLS binary array for the piece occupancies of the active player
        T[2] is a ROWS x COLS binary array for the piece occupancies of the inactive player
        T[3] is a ROWS x COLS array of identical numbers equal to the active players number of captures
        T[4] is a ROWS x COLS array of identical numbers equal to the inactive players number of captures
        
        Notes:
        We dont need to explicitly pass color information. Due to the move symettry of pente, we can instead encode this
            in the field orderings as active and inacvtive player.
        Layers 3 and 4 may seem wasteful. In practice, the first layers of the network will be convolutional.
        By uniformly setting the entire layer, we guarantee this information passes uniformly to subsequent layers of the network.
        Furthermore, convolutional layers will not suffer heavy computations due to this "redundant" information so it isnt actually that bad.
        """ 
        t = np.array((4, ROWS, COLS), dtype=np.float32)
        if self.is_white_:
            t[0, :, :] = self.w_occs_
            t[1, :, :] = self.b_occs_
            t[2, :, :] = self.w_caps_
            t[3, :, :] = self.b_caps_
            return t
        
        t[0, :, :] = self.b_occs_
        t[1, :, :] = self.w_occs_
        t[2, :, :] = self.b_caps_
        t[3, :, :] = self.w_caps_
        return t

    def apply_action(self, action_id):
        """
        action_id MUST be a legal move.
        For performance reasons, we dont implement a safegaurd that checks this.
        """
        chunk, chunk_rem = _action_id_2_chunk(action_id)
        self._fill(chunk, chunk_rem, self.is_white_)
        self.num_pieces += 1
        row, col = action_id_2_coord(action_id)
        self._update_captures(row, col)
        if self._caps() == 5 or self._is_con5(row, col):
            if self.is_white_:
                self.game_state_= State.White
            else:
                self.game_state_ = State.Black
        elif self.num_pieces == ROWS*COLS:
            self.game_state_ = State.Draw
        self.is_white_ = not self.is_white_


    def get_legal_moves(self):
        """
        Returns mask of the action space A as array of type np.uint8. Has |A| elements.
        Each element is 0 or 1. 0 means that action is illegal. 1 means legal
        """
        if self.is_terminal():
            r = np.array([], dtype=np.uint8)
        else:
            l = (~self.w_occs_ & ~self.b_occs_ & BRD_MASK).view(np.uint8)
            r = np.unpackbits(l, bitorder="little")
        return r

    def is_terminal(self):
        return self.game_state_ != State.Running
    

    def get_terminal_value(self, is_white):
        """
        Assumes is_terminal() == True.
        Returns the score {-1, 0, 1} relative to the winner.
        -1 if provided provided player lost. 0 for draw. 1 for win
        """
        if (self.is_white_ and self.game_state_ == State.White) \
                or (self.is_black_ and self.game_state_ == State.Black):
            return 1
        elif self.game_state_ == State.Draw:
            return 0
        else:
            return -1



    def clone(self):
        """Perform deep copy"""
        g = GameState()
        g.w_occs_ = self.w_occs_.copy()
        g.b_occs_ = self.b_occs_.copy()
        g.w_caps_ = self.w_caps_.copy()
        g.b_caps_ = self.b_caps_.copy()
        g.num_pieces = self.num_pieces
        g.is_white_ = self.is_white_
        g.game_state_ = self.game_state_
        return g
    

    def play(self, row:np.uint8, col:np.uint8):
        if not self.is_terminal():
            self.apply_action(_coord_2_action_id(row,col))


    def play_safe(self, row:np.uint8, col:np.uint8) -> bool:
        """ checks if move is legal. Returns True on success, false on failure."""
        action = _coord_2_action_id(row,col)
        if not self.is_terminal() and self.get_legal_moves()[action]:
            self.apply_action(action)
            return True
        return False    


    def is_white(self):
        return self.is_white_

    def _caps(self):
        if self.is_white_:
            return self.w_caps_
        return self.b_caps_

    def _occs(self):
        if self.is_white_:
            return self.w_occs_
        return self.b_occs_


    def _is_con5(self, row, col):
        count = 1
        for i in range(1, 5): # Vertical
            if _coord_inbounds(row-i, col) and self._is_occ_coord(row-i, col, self.is_white_):
                count += 1
            else:
                break
        for i in range(1, 5):   
            if _coord_inbounds(row+i, col) and self._is_occ_coord(row+i, col, self.is_white_):
                count += 1
            else:
                break
        if count >= 5:
            return True
        
        count = 1
        for i in range(1, 5): # Horizontal
            if _coord_inbounds(row, col-i) and self._is_occ_coord(row, col-i, self.is_white_):
                count += 1
            else:
                break
        for i in range(1, 5):   
            if _coord_inbounds(row, col+i) and self._is_occ_coord(row, col+i, self.is_white_):
                count += 1
            else:
                break
        if count >= 5:
            return True
        
        count = 1
        for i in range(1, 5): # Diag \
            if _coord_inbounds(row-i, col-i) and self._is_occ_coord(row-i, col-i, self.is_white_):
                count += 1
            else:
                break
        for i in range(1, 5):   
            if _coord_inbounds(row+i, col+i) and self._is_occ_coord(row+i, col+i, self.is_white_):
                count += 1
            else:
                break
        if count >= 5:
            return True
        
        count = 1
        for i in range(1, 4): # Diag /
            if _coord_inbounds(row-i, col+i) and self._is_occ_coord(row-i, col+i, self.is_white_):
                count += 1
            else:
                break
        for i in range(1, 4):   
            if _coord_inbounds(row+i, col-i) and self._is_occ_coord(row+i, col-i, self.is_white_):
                count += 1
            else:
                break
        if count >= 5:
            return True
        return False


    def _update_captures(self, row, col):
        if row >= 3:    # Vertical down
            if self._is_occ_coord(row-3, col, self.is_white_) \
                    and self._is_occ_coord(row-2, col, not self.is_white_) \
                    and self._is_occ_coord(row-1, col, not self.is_white_):
                self._del_coord(row-2, col, not self.is_white_)
                self._del_coord(row-1, col, not self.is_white_)
                self.num_pieces -=2
                self._adj_cap(1, self.is_white_) 
        
        if row <= ROWS-4: # Vertical up
            if self._is_occ_coord(row+3, col, self.is_white_) \
                    and self._is_occ_coord(row+2, col, not self.is_white_) \
                    and self._is_occ_coord(row+1, col, not self.is_white_):
                self._del_coord(row+2, col, not self.is_white_)
                self._del_coord(row+1, col, not self.is_white_)
                self.num_pieces -=2
                self._adj_cap(1, self.is_white_) 

        if col >= 3: # Horizontal left
            if self._is_occ_coord(row, col-3, self.is_white_) \
                    and self._is_occ_coord(row, col-2, not self.is_white_) \
                    and self._is_occ_coord(row, col-1, not self.is_white_):
                self._del_coord(row, col-2, not self.is_white_)
                self._del_coord(row, col-1, not self.is_white_)
                self.num_pieces -=2
                self._adj_cap(1, self.is_white_) 
        
        if col <= COLS-4: # Horizontal right
            if self._is_occ_coord(row, col+3, self.is_white_) \
                    and self._is_occ_coord(row, col+2, not self.is_white_) \
                    and self._is_occ_coord(row, col+1, not self.is_white_):
                self._del_coord(row, col+2, not self.is_white_)
                self._del_coord(row, col+1, not self.is_white_)
                self.num_pieces -=2
                self._adj_cap(1, self.is_white_) 
        
        if row >= 3 and col >= 3: # Diagonal top-left
            if self._is_occ_coord(row-3, col-3, self.is_white_) \
                    and self._is_occ_coord(row-2, col-2, not self.is_white_) \
                    and self._is_occ_coord(row-1, col-1, not self.is_white_):
                self._del_coord(row-2, col-2, not self.is_white_)
                self._del_coord(row-1, col-1, not self.is_white_)
                self.num_pieces -=2
                self._adj_cap(1, self.is_white_) 
        
        if row>= 3 and col <= COLS-4: # Diagonal top-right
            if self._is_occ_coord(row-3, col+3, self.is_white_) \
                    and self._is_occ_coord(row-2, col+2, not self.is_white_) \
                    and self._is_occ_coord(row-1, col+1, not self.is_white_):
                self._del_coord(row-2, col+2, not self.is_white_)
                self._del_coord(row-1, col+1, not self.is_white_)
                self.num_pieces -=2
                self._adj_cap(1, self.is_white_) 
        
        if row <= ROWS-4 and col >= 3: # Diagonal bottom-left
            if self._is_occ_coord(row+3, col-3, self.is_white_) \
                    and self._is_occ_coord(row+2, col-2, not self.is_white_) \
                    and self._is_occ_coord(row+1, col-1, not self.is_white_):
                self._del_coord(row+2, col-2, not self.is_white_)
                self._del_coord(row+1, col-1, not self.is_white_)
                self.num_pieces -=2
                self._adj_cap(1, self.is_white_) 

        if row <= ROWS-4 and col <= COLS-4:     # Diagonal bottom-right
            if self._is_occ_coord(row+3, col+3, self.is_white_) \
                    and self._is_occ_coord(row+2, col+2, not self.is_white_) \
                    and self._is_occ_coord(row+1, col+1, not self.is_white_):
                self._del_coord(row+2, col+2, not self.is_white_)
                self._del_coord(row+1, col+1, not self.is_white_)
                self.num_pieces -=2
                self._adj_cap(1, self.is_white_) 

                
    def _adj_cap(self, delta:int, is_white):
        """ Adjust the tally for the number of captures """
        if is_white:
            self.w_caps_ += delta
            return self.w_caps_
        else:
            self.b_caps_ += delta
            self.b_caps_

    def _fill(self, chunk, chunk_rem, is_white:bool):
        if is_white:
            self.w_occs_[chunk] |=  np.uint64(1) << chunk_rem
        else:
            self.b_occs_[chunk] |=  np.uint64(1) << chunk_rem


    def _del_coord(self, row, col, is_white):
        """ Clears occupancy bit """
        chunk, chunk_rem = _coord_2_chunk(row, col)
        self._del_chunk(chunk, chunk_rem, is_white)


    def _del_chunk(self, chunk, chunk_rem, is_white:bool):
            if is_white:
                self.w_occs_[chunk] &=  ~(np.uint64(1) << chunk_rem)
            else:
                self.b_occs_[chunk] &=  ~(np.uint64(1) << chunk_rem)
    
            
    def _is_occ_coord(self, row, col, is_white):
        chunk, chunk_rem = _coord_2_chunk(row, col)
        return self._is_occ_chunk(chunk, chunk_rem, is_white)


    def _is_occ_chunk(self, chunk, chunk_rem, is_white:bool) -> bool:
        if is_white:
            x = (self.w_occs_[chunk] >> chunk_rem) & np.uint64(1)
            return x == 1
        else:
            x = (self.b_occs_[chunk] >> chunk_rem) & np.uint64(1)
            return x == 1

    def __str__(self):
        s = "  "
        for c in range(0,COLS):
            s += f"{c:02d} "
        s += "\n"
        for r in range(0,ROWS):
            s += f"{r:02d} "
            for c in range(0,COLS):
                chunk, chunk_rem = _coord_2_chunk(r,c)
                if self._is_occ_chunk(chunk, chunk_rem, is_white=True):
                    s += "W  "
                elif self._is_occ_chunk(chunk, chunk_rem, is_white=False):
                    s += "B  "
                else:
                    s += "•  "
            s += "\n"
        s += f"White Captures: {self.w_caps_}      Black Captures: {self.b_caps_}"
        return s
        
def action_space_size()->int:
    return ROWS*COLS
    


if __name__ == "__main__":
    brd = GameState()
    while(True):
        print(brd)
        player_str = {True: "White", False:"Black"}
        user_in = input(f"Enter {player_str[brd.is_white()]} move as row, col: ")
        coord = user_in.split(",")
        if len(coord) == 2:
            row = int(coord[0])
            col = int(coord[1])
            if brd.play_safe(row, col):
                if brd.is_terminal():
                    print(f"Game over. {player_str[brd.is_white()]} Win!")
                    break
            else:
                print("Invalid move, try again")
                continue

        else:
            print("Unable to parse input, try again")
            continue
        
            