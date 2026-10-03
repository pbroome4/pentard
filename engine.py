from dataclasses import dataclass
import numpy as np


# Board is a 19 x 19. We will store this as a bitboard using u64 ints.
# This requires 7 u64's to store occupancies for each player. 
ROWS = 19      
COLS = 19       
BRD_LEN = 7




class BoardState:
    def __init__(self):
        self.w_occs = np.zeros((ROWS,COLS), dtype=np.uint64)
        self.w_caps = np.uint8(0)
        self.b_occs = np.zeros((ROWS, COLS), dtype=np.uint64)
        self.b_caps = np.uint8(0)
        self.game_over: bool = False


    def play(self, row:np.uint8, col:np.uint8, is_white:bool):
        chunk, chunk_rem = BoardState._coord(row, col)
        self._fill(chunk, chunk_rem, is_white)



    def _fill(self, chunk, chunk_rem, is_white:bool):
        if is_white:
            self.w_occs[chunk] |=  np.uint64(1) << chunk_rem
        else:
            self.b_occs[chunk] |=  np.uint64(1) << chunk_rem


    def _del(self, chunk, chunk_rem, is_white:bool):
            if is_white:
                self.w_occs[chunk] &=  ~(np.uint64(1) << chunk_rem)
            else:
                self.b_occs[chunk] &=  ~(np.uint64(1) << chunk_rem)
    
            

    def _get_occ(self, chunk, chunk_rem, is_white:bool) -> bool:
        if is_white:
            x = (self.w_occs[chunk] >> chunk_rem) & np.uint64(1)
            return x[0]
        else:
            x = (self.b_occs[chunk] >> chunk_rem) & np.uint64(1)
            return x[0]

    def __str__(self):
        s = "  "
        for c in range(0,COLS):
            s += f"{c:02d} "
        s += "\n"
        for r in range(0,ROWS):
            s += f"{c:02d} "
            for c in range(0,COLS):
                chunk, chunk_rem = self._coord(r,c)
                if self._get_occ(chunk, chunk_rem, is_white=True):
                    s += "W  "
                elif self._get_occ(chunk, chunk_rem, is_white=False):
                    s += "B  "
                else:
                    s += "•  "
            s += "\n"
        
        return s
        

    @staticmethod
    def _coord(row: np.uint8, col:np.uint8):
        bit_pos:np.uint8 = row*COLS + col
        occ_chunk = bit_pos // 64
        occ_chunk_rem = bit_pos - occ_chunk * 64
        return (occ_chunk, occ_chunk_rem)
    

if __name__ == "__main__":
    brd = BoardState()
    is_white = True
    while(True):
        print(brd)
        player_str = {True: "White", False:"Black"}
        user_in = input(f"Enter {player_str[is_white]} move as row, col: ")
        coord = user_in.split(",")
        if len(coord) == 2:
            row = int(coord[0])
            col = int(coord[1])
            brd.play(row,col, is_white)
            is_white = not is_white
        else:
            print("Unable to parse input, try again")
            continue
        
            