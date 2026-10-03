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


    def play_safe(self, row:np.uint8, col:np.uint8, is_white:bool):
        chunk, chunk_rem = BoardState._coord_2_chunk(row, col)
        if  not self._is_occ_chunk(chunk, chunk_rem, is_white = True) \
                and not self._is_occ_chunk(chunk, chunk_rem, is_white = False):
            self.play(row, col, is_white)
            return True
        return False
    
        
    def play(self, row:np.uint8, col:np.uint8, is_white:bool):
        chunk, chunk_rem = BoardState._coord_2_chunk(row, col)
        self._fill(chunk, chunk_rem, is_white)
        self._update_captures(row, col, is_white)
        is_con5 = self._is_con5(row, col, is_white)
        if self.b_caps == 5 or self.w_caps == 5 or is_con5:
            self.game_over = True


    def is_game_over(self):
        return self.game_over


    def _is_con5(self, row, col, is_white):
        count = 1
        for i in range(1, 5): # Vertical
            if self._coord_inbounds(row-i, col) and self._is_occ_coord(row-i, col, is_white):
                count += 1
            else:
                break
        for i in range(1, 5):   
            if self._coord_inbounds(row+i, col) and self._is_occ_coord(row+i, col, is_white):
                count += 1
            else:
                break
        if count >= 5:
            return True
        
        count = 1
        for i in range(1, 5): # Horizontal
            if self._coord_inbounds(row, col-i) and self._is_occ_coord(row, col-i, is_white):
                count += 1
            else:
                break
        for i in range(1, 5):   
            if self._coord_inbounds(row, col+i) and self._is_occ_coord(row, col+i, is_white):
                count += 1
            else:
                break
        print(f"horiz count: {count}")
        if count >= 5:
            return True
        
        count = 1
        for i in range(1, 5): # Diag \
            if self._coord_inbounds(row-i, col-i) and self._is_occ_coord(row-i, col-i, is_white):
                count += 1
            else:
                break
        for i in range(1, 5):   
            if self._coord_inbounds(row+i, col+i) and self._is_occ_coord(row+i, col+i, is_white):
                count += 1
            else:
                break
        if count >= 5:
            return True
        
        count = 1
        for i in range(1, 4): # Diag /
            if self._coord_inbounds(row-i, col+i) and self._is_occ_coord(row-i, col+i, is_white):
                count += 1
            else:
                break
        for i in range(1, 4):   
            if self._coord_inbounds(row+i, col-i) and self._is_occ_coord(row+i, col-i, is_white):
                count += 1
            else:
                break
        if count >= 5:
            return True
        return False


    def _update_captures(self, row, col, is_white):
        if row >= 3:    # Vertical down
            if self._is_occ_coord(row-3, col, is_white) \
                    and self._is_occ_coord(row-2, col, not is_white) \
                    and self._is_occ_coord(row-1, col, not is_white):
                self._del_coord(row-2, col, not is_white)
                self._del_coord(row-1, col, not is_white)
                self._adj_cap(1, is_white) 
        
        if row <= ROWS-4: # Vertical up
            if self._is_occ_coord(row+3, col, is_white) \
                    and self._is_occ_coord(row+2, col, not is_white) \
                    and self._is_occ_coord(row+1, col, not is_white):
                self._del_coord(row+2, col, not is_white)
                self._del_coord(row+1, col, not is_white)
                self._adj_cap(1, is_white) 

        if col >= 3: # Horizontal left
            if self._is_occ_coord(row, col-3, is_white) \
                    and self._is_occ_coord(row, col-2, not is_white) \
                    and self._is_occ_coord(row, col-1, not is_white):
                self._del_coord(row, col-2, not is_white)
                self._del_coord(row, col-1, not is_white)
                self._adj_cap(1, is_white) 
        
        if col <= COLS-4: # Horizontal right
            if self._is_occ_coord(row, col+3, is_white) \
                    and self._is_occ_coord(row, col+2, not is_white) \
                    and self._is_occ_coord(row, col+1, not is_white):
                self._del_coord(row, col+2, not is_white)
                self._del_coord(row, col+1, not is_white)
                self._adj_cap(1, is_white) 
        
        if row >= 3 and col >= 3: # Diagonal top-left
            if self._is_occ_coord(row-3, col-3, is_white) \
                    and self._is_occ_coord(row-2, col-2, not is_white) \
                    and self._is_occ_coord(row-1, col-1, not is_white):
                self._del_coord(row-2, col-2, not is_white)
                self._del_coord(row-1, col-1, not is_white)
                self._adj_cap(1, is_white) 
        
        if row>= 3 and col <= COLS-4: # Diagonal top-right
            if self._is_occ_coord(row-3, col+3, is_white) \
                    and self._is_occ_coord(row-2, col+2, not is_white) \
                    and self._is_occ_coord(row-1, col+1, not is_white):
                self._del_coord(row-2, col+2, not is_white)
                self._del_coord(row-1, col+1, not is_white)
                self._adj_cap(1, is_white) 
        
        if row <= ROWS-4 and col >= 3: # Diagonal bottom-left
            if self._is_occ_coord(row+3, col-3, is_white) \
                    and self._is_occ_coord(row+2, col-2, not is_white) \
                    and self._is_occ_coord(row+1, col-1, not is_white):
                self._del_coord(row+2, col-2, not is_white)
                self._del_coord(row+1, col-1, not is_white)
                self._adj_cap(1, is_white) 

        if row <= ROWS-4 and col <= COLS-4:     # Diagonal bottom-right
            if self._is_occ_coord(row+3, col+3, is_white) \
                    and self._is_occ_coord(row+2, col+2, not is_white) \
                    and self._is_occ_coord(row+1, col+1, not is_white):
                self._del_coord(row+2, col+2, not is_white)
                self._del_coord(row+1, col+1, not is_white)
                self._adj_cap(1, is_white) 

                
    def _adj_cap(self, delta:int, is_white):
        if is_white:
            self.w_caps += delta
            return self.w_caps
        else:
            self.b_caps += delta
            self.b_caps

    def _fill(self, chunk, chunk_rem, is_white:bool):
        if is_white:
            self.w_occs[chunk] |=  np.uint64(1) << chunk_rem
        else:
            self.b_occs[chunk] |=  np.uint64(1) << chunk_rem


    def _del_coord(self, row, col, is_white):
        chunk, chunk_rem = self._coord_2_chunk(row, col)
        self._del_chunk(chunk, chunk_rem, is_white)


    def _del_chunk(self, chunk, chunk_rem, is_white:bool):
            print(f"deleting chunk:{chunk}  chunk_rem:{chunk_rem}")
            if is_white:
                self.w_occs[chunk] &=  ~(np.uint64(1) << chunk_rem)
            else:
                self.b_occs[chunk] &=  ~(np.uint64(1) << chunk_rem)
    
            
    def _is_occ_coord(self, row, col, is_white):
        chunk, chunk_rem = self._coord_2_chunk(row, col)
        return self._is_occ_chunk(chunk, chunk_rem, is_white)


    def _is_occ_chunk(self, chunk, chunk_rem, is_white:bool) -> bool:
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
            s += f"{r:02d} "
            for c in range(0,COLS):
                chunk, chunk_rem = self._coord_2_chunk(r,c)
                if self._is_occ_chunk(chunk, chunk_rem, is_white=True):
                    s += "W  "
                elif self._is_occ_chunk(chunk, chunk_rem, is_white=False):
                    s += "B  "
                else:
                    s += "•  "
            s += "\n"
        s += f"White Captures: {self.w_caps}      Black Captures: {self.b_caps}"
        return s
        

    @staticmethod
    def _coord_2_chunk(row: np.uint8, col:np.uint8):
        bit_pos:np.uint8 = row*COLS + col
        occ_chunk = bit_pos // 64
        occ_chunk_rem = bit_pos - occ_chunk * 64
        return (occ_chunk, occ_chunk_rem)

    @staticmethod
    def _chunk_2_coord(chunk, chunk_rem):
        bit_pos = chunk * 64 + chunk_rem
        row = bit_pos / COLS
        col = bit_pos - row*COLS
        return (row,col)

    @staticmethod
    def _coord_inbounds(row, col):
        return row >= 0 and row < ROWS and col >= 0 and col < COLS

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
            if brd.play_safe(row,col, is_white):
                if brd.is_game_over():
                    print(f"Game over. {player_str[is_white]} Win!")
                    break
                is_white = not is_white
            else:
                print("Invalid move, try agains")
                continue

        else:
            print("Unable to parse input, try again")
            continue
        
            