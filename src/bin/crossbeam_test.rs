use crossbeam::channel;
use std::thread;

// Define your struct data
#[derive(Debug)]
struct MathResult {
    thread_id: usize,
    data: Vec<f64>,
}

// Your heavy math function
fn heavy_math_function(id: usize) -> Vec<f64> {
    // Simulate heavy computation generating a vector of data
    let mut result = Vec::with_capacity(1000);
    for i in 0..1000 {
        result.push((id as f64 * 100.0) + (i as f64 * 0.5).sin());
    }
    result
}

fn main() {
    // Create an unbounded channel. 
    // `s` is the sender, `r` is the receiver.
    let (s, r) = channel::unbounded::<MathResult>();

    crossbeam::scope(|scope| {
        // 1. Spawn the Reader/Consumer Thread
        let reader_sender = s.clone(); // Keep a sender alive or drop it appropriately
        // Drop our local reference to `s` so the reader knows when producers are done.
        // (Only the spawned producer threads will hold senders).
        drop(s);
        scope.spawn(move |_| {
            println!("Reader thread started, waiting for data...");
            
            // The `for` loop on receiver `r` automatically blocks and waits for messages, 
            // and terminates gracefully once ALL sender clones are dropped.
            for result in r {
                println!("Reader consumed data from thread {}: length = {}", 
                    result.thread_id, result.data.len());
                // Process or aggregate your data here
            }
            
            println!("Reader thread finished.");
        });

        

        // 2. Spawn Multiple Producer Threads (e.g., 4 threads)
        for thread_id in 0..4 {
            let thread_sender = reader_sender.clone();
            
            scope.spawn(move |_| {
                // Run the heavy math function
                let computed_vector = heavy_math_function(thread_id);

                // Package it into your struct
                let payload = MathResult {
                    thread_id,
                    data: computed_vector,
                };

                // Send (append) the data to the common pipeline
                thread_sender.send(payload).unwrap();
                println!("Producer thread {} finished and sent its data.", thread_id);
            });
        }
        
        // When the scope block ends, all producer threads complete 
        // and their `thread_sender` handles are automatically dropped.
    }).unwrap();

    println!("All processing complete!");
}