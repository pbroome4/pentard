
use std::io::{self, Write}; // Bring the I/O traits into scope
use pentard::engine::*;


fn parse_coord(s: &str) -> Result<(usize, usize), String> {
    let mut parts = s.trim().split(',');
    if let (Some(row_str), Some(col_str)) = (parts.next(), parts.next()) {
        let x: Result<usize, _> = row_str.trim().parse();
        let y: Result<usize, _> = col_str.trim().parse();
        match (x, y) {
            (Ok(row), Ok(col)) => {
                return Ok((row, col));
            }
            _ =>{
                return Err(String::from("Invalid format"));
            } 
        }
    }
    return Err(String::from("Invalid format"));
}

fn read_move() -> Result<(usize, usize), String>{
    let mut input_buffer = String::new();
    io::stdin().read_line(&mut input_buffer).unwrap();
    match parse_coord(&input_buffer){
        Ok((r,c)) => {
            return Ok((r,c));
        }
        Err(s) =>{
            return Err(s);
        } 
    }
}

fn main(){
    let mut brd = PenteState::new();
    loop{
        print!("{}", brd);
        let player: &str = brd.active_plyr_str();
        print!("Enter move for {player} as row, col: ");
        io::stdout().flush().unwrap();
        if let Ok((row, col)) = read_move() {
            if brd.play(row, col){
                if brd.is_terminal(){
                    break;
                }
            }else{
                print!("Illegal Move");
                continue;
            }
        }else{
            print!("Failed to read move")
        }
    }
    print!("{}", brd);
    print!("Game Over!");
}