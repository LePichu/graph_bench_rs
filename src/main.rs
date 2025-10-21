use std::env;
use std::process;

mod algo;
mod bench;
mod parser;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: {} <file_path> <start_node> <end_node> [runs]",
            args[0]
        );
        process::exit(1);
    }

    let file_path = &args[1];
    let start_node = &args[2];
    let end_node = &args[3];

    let runs = if args.len() > 4 {
        args[4].parse().unwrap_or(20)
    } else {
        20
    };

    println!(
        "Running benchmark with {} runs for graph '{}' from '{}' to '{}'.\n",
        runs, file_path, start_node, end_node
    );

    let graph = match parser::parse_graph(file_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error parsing graph file: {}", e);
            process::exit(1);
        }
    };

    // Quick path demonstration before the full benchmark
    match algo::dijkstra(&graph, start_node, end_node) {
        Some((path, cost)) => {
            println!("Dijkstra Demo Path: {:?}, Cost: {}\n", path, cost);
        }
        None => {
            println!(
                "Dijkstra Demo: No path found from {} to {}.\n",
                start_node, end_node
            );
        }
    }

    bench::run_and_print_benchmarks(&graph, start_node, end_node, runs);
}
