
use std::io::{self, Write}; // Bring the I/O traits into scope
use std::str::FromStr;
use tracing::{debug, info};
use tracing::Level;
use tracing_subscriber;
use clap::Parser;

use pentard::pente::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args{
    /// trace, debug, info, warning, or error. Default info
    #[arg(short, long, default_value = "info")]
    log_level: String,
    
    /// Number of nodes to explore. Default 1,000
    #[arg(short, long, default_value = "1000")]
    n: u64,
}


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
    let args = Args::parse();
    println!("{}", args.log_level);
    tracing_subscriber::fmt()
        .with_max_level(Level::from_str(&args.log_level).unwrap()) // Set max level: TRACE, DEBUG, INFO, WARN, ERROR, or OFF
        .init();

    let mut brd = PenteState::new();
    loop{
        info!("\n{}", brd);
        //mcts(&brd, args.n);
        let player: &str = brd.active_plyr_str();
        info!("Enter move for {player} as row, col: ");
        io::stdout().flush().unwrap();
        if let Ok((row, col)) = read_move() {
            if brd.play(row, col){
                if brd.is_terminal(){
                    break;
                }
            }else{
                info!("Illegal Move");
                continue;
            }
        }else{
            info!("Failed to read move")
        }
    }
    info!("{}", brd);
    info!("Game Over!");
}