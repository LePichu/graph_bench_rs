use crate::algo;
use crate::parser::Graph;
use prettytable::{row, Cell, Row, Table};
use std::time::{Duration, Instant};

struct BenchResult {
    name: String,
    min_ms: f64,
    max_ms: f64,
    average_ms: f64,
}

pub fn run_and_print_benchmarks(graph: &Graph, start: &str, end: &str, runs: usize) {
    let mut results = Vec::new();

    let algorithms: Vec<(&str, fn(&Graph, &str, &str) -> Option<(Vec<String>, u32)>)> = vec![
        ("Dijkstra", algo::dijkstra),
        ("Bellman-Ford", algo::bellman_ford),
        ("Hybrid SSSP", algo::hybrid_sssp),
    ];

    for (name, algorithm_fn) in algorithms {
        let mut durations: Vec<Duration> = Vec::with_capacity(runs);
        for _ in 0..runs {
            let start_time = Instant::now();
            algorithm_fn(graph, start, end);
            let duration = start_time.elapsed();
            durations.push(duration);
        }

        let total_duration: Duration = durations.iter().sum();
        let average_duration = if runs > 0 {
            total_duration / (runs as u32)
        } else {
            Duration::new(0, 0)
        };

        let zero_duration = Duration::new(0, 0);
        let min_duration = durations.iter().min().unwrap_or(&zero_duration);
        let max_duration = durations.iter().max().unwrap_or(&zero_duration);

        results.push(BenchResult {
            name: name.to_string(),
            min_ms: min_duration.as_secs_f64() * 1000.0,
            max_ms: max_duration.as_secs_f64() * 1000.0,
            average_ms: average_duration.as_secs_f64() * 1000.0,
        });
    }

    let mut table = Table::new();
    table.add_row(row![b => "#", "Start Node", "End Node", "Algorithm", "Min (ms)", "Max (ms)", "Average (ms)"]);

    for (i, result) in results.iter().enumerate() {
        table.add_row(Row::new(vec![
            Cell::new(&(i + 1).to_string()),
            Cell::new(start),
            Cell::new(end),
            Cell::new(&result.name),
            Cell::new(&format!("{:.6}", result.min_ms)),
            Cell::new(&format!("{:.6}", result.max_ms)),
            Cell::new(&format!("{:.6}", result.average_ms)),
        ]));
    }

    table.printstd();

    if let Some(fastest) = results
        .iter()
        .min_by(|a, b| a.average_ms.partial_cmp(&b.average_ms).unwrap())
    {
        if let Some(slowest) = results
            .iter()
            .max_by(|a, b| a.average_ms.partial_cmp(&b.average_ms).unwrap())
        {
            println!(
                "\nFastest: {} ({:.6}ms avg) | Slowest: {} ({:.6}ms avg)",
                fastest.name, fastest.average_ms, slowest.name, slowest.average_ms
            );
        }
    }
}
