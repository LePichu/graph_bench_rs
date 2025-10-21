use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::fs;

#[derive(Debug, Deserialize, Clone)]
pub struct Edge {
    pub node: String,
    pub weight: u32,
}

pub type Graph = HashMap<String, Vec<Edge>>;

pub fn parse_graph(file_path: &str) -> Result<Graph, Box<dyn Error>> {
    let content = fs::read_to_string(file_path)?;
    let graph: Graph = serde_json::from_str(&content)?;
    Ok(graph)
}
