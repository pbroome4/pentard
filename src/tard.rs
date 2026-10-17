use tch::{nn};

////////////////////////////////////////
//       ### Thread 1 ####
//   1) Sample training data from replay buffer
//   2) update network
//
//       #### Thread 2 ###
//   1) Create batch of games at starting position (start with batch size 1)
//   2) Call library to run MCTS
//       2.1) For each game, Run N(=800, 1600 etc?) MCTS runouts
//            This MCTS requires network prior and value output for PUCT
//       2.2) At the end of N MCTS runouts, make best move found for the game. 
//       2.3) repeat until all games end
//       2.4) Return training data IE:
//            For each state in each game, return (state, win/loss/draw),
//            network value, actual game terminal value(win/loss).
//   3) Add training data to a replay buffer
//   4) Resync with thread1 model which has had aditional training
//      

struct Tard{
    conv1: nn::Conv2D,
}

