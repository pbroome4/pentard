use tracing::{Level, debug, info};
use tracing_subscriber;
use pentard;

fn main(){
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG) // Set max level: TRACE, DEBUG, INFO, WARN, ERROR, or OFF
        .init();
    pentard::tard::train_tard();
}