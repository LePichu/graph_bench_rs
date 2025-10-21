# Graph Algorithm Bench
A Simple Benchmark between Dijkstra, Bellman-Ford and the newly proposed Hybrid Method by the researchers at Stanford and Tsinghua University which provides a faster way to find the shortest path between 2 nodes on larger graphs.

## Instructions
- Install Rust from [`rustup`](https://rustup.rs).
- Clone this Repository.
- Run `cargo run -- <GRAPH> <START_NODE> <END_NOTE>` at root, example command: `cargo run --release -- ./test/graph-5000.json N0 N99`.

## License
This code is licensed under the MIT License, it uses the original research paper as a reference point to implement the algorithm in Rust, please look at [LICENSE](./LICENSE) for more information.
