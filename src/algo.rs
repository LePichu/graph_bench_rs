use crate::parser::Graph;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct State {
    cost: u32,
    node_idx: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn dijkstra(graph: &Graph, start_node: &str, end_node: &str) -> Option<(Vec<String>, u32)> {
    let (node_map, rev_node_map) = build_node_maps(graph);
    let start_idx = match node_map.get(start_node) {
        Some(&idx) => idx,
        None => return None,
    };
    let end_idx = match node_map.get(end_node) {
        Some(&idx) => idx,
        None => return None,
    };

    let mut dist: Vec<_> = (0..node_map.len()).map(|_| u32::MAX).collect();
    let mut prev = vec![None; node_map.len()];
    let mut heap = BinaryHeap::new();

    dist[start_idx] = 0;
    heap.push(State {
        cost: 0,
        node_idx: start_idx,
    });

    while let Some(State { cost, node_idx }) = heap.pop() {
        if cost > dist[node_idx] {
            continue;
        }

        let node_name = rev_node_map.get(&node_idx).unwrap();
        if let Some(edges) = graph.get(node_name) {
            for edge in edges {
                if let Some(&neighbor_idx) = node_map.get(&edge.node) {
                    let next = State {
                        cost: cost + edge.weight,
                        node_idx: neighbor_idx,
                    };
                    if next.cost < dist[next.node_idx] {
                        heap.push(next);
                        dist[next.node_idx] = next.cost;
                        prev[next.node_idx] = Some(node_idx);
                    }
                }
            }
        }
    }

    if dist[end_idx] != u32::MAX {
        Some((
            reconstruct_path(start_idx, end_idx, &prev, &rev_node_map),
            dist[end_idx],
        ))
    } else {
        None
    }
}

pub fn bellman_ford(graph: &Graph, start_node: &str, end_node: &str) -> Option<(Vec<String>, u32)> {
    let (node_map, rev_node_map) = build_node_maps(graph);
    let start_idx = match node_map.get(start_node) {
        Some(&idx) => idx,
        None => return None,
    };
    let end_idx = match node_map.get(end_node) {
        Some(&idx) => idx,
        None => return None,
    };

    let num_nodes = node_map.len();
    let mut dist = vec![u32::MAX; num_nodes];
    let mut prev = vec![None; num_nodes];
    dist[start_idx] = 0;

    for _ in 1..num_nodes {
        for (u_name, edges) in graph {
            if let Some(&u_idx) = node_map.get(u_name) {
                if dist[u_idx] == u32::MAX {
                    continue;
                }
                for edge in edges {
                    if let Some(&v_idx) = node_map.get(&edge.node) {
                        if dist[u_idx] + edge.weight < dist[v_idx] {
                            dist[v_idx] = dist[u_idx] + edge.weight;
                            prev[v_idx] = Some(u_idx);
                        }
                    }
                }
            }
        }
    }

    if dist[end_idx] != u32::MAX {
        Some((
            reconstruct_path(start_idx, end_idx, &prev, &rev_node_map),
            dist[end_idx],
        ))
    } else {
        None
    }
}

pub fn hybrid_sssp(graph: &Graph, start_node: &str, end_node: &str) -> Option<(Vec<String>, u32)> {
    let n = graph.len() as f64;
    if n == 0.0 {
        return None;
    }

    let k = n.log(2.0).powf(1.0 / 3.0).floor() as usize;
    let t = n.log(2.0).powf(2.0 / 3.0).floor() as usize;
    let l_max = if t > 0 {
        (n.log2() / t as f64).ceil() as usize
    } else {
        1
    };

    let (node_map, rev_node_map) = build_node_maps(graph);

    let mut d_hat = vec![u32::MAX; node_map.len()];
    let mut pred = vec![None; node_map.len()];

    let start_idx = match node_map.get(start_node) {
        Some(&idx) => idx,
        None => return None,
    };
    let end_idx = match node_map.get(end_node) {
        Some(&idx) => idx,
        None => return None,
    };
    d_hat[start_idx] = 0;

    let s = HashSet::from([start_idx]);
    bmssp(
        l_max,
        u32::MAX,
        s,
        graph,
        &node_map,
        k,
        t,
        &mut d_hat,
        &mut pred,
    );

    if d_hat[end_idx] != u32::MAX {
        Some((
            reconstruct_path(start_idx, end_idx, &pred, &rev_node_map),
            d_hat[end_idx],
        ))
    } else {
        None
    }
}

fn bmssp(
    l: usize,
    b: u32,
    s: HashSet<usize>,
    graph: &Graph,
    node_map: &HashMap<String, usize>,
    k: usize,
    t: usize,
    d_hat: &mut Vec<u32>,
    pred: &mut Vec<Option<usize>>,
) -> (u32, HashSet<usize>) {
    if l == 0 {
        return base_case(
            b,
            s.into_iter().next().unwrap(),
            graph,
            node_map,
            k,
            d_hat,
            pred,
        );
    }

    let (p, w) = find_pivots(b, &s, graph, node_map, k, d_hat, pred);
    let mut d: BinaryHeap<_> = p
        .iter()
        .map(|&idx| State {
            cost: d_hat[idx],
            node_idx: idx,
        })
        .collect();
    let mut u_total = HashSet::new();

    let m = 2_usize.pow(((l - 1) * t) as u32);

    while !d.is_empty() && u_total.len() < k * 2_usize.pow((l * t) as u32) {
        let mut s_i = HashSet::new();
        let mut b_i = b;
        let mut temp_pulled = Vec::new();

        while s_i.len() < m && !d.is_empty() {
            let state = d.pop().unwrap();
            s_i.insert(state.node_idx);
            temp_pulled.push(state);
        }
        if let Some(next_state) = d.peek() {
            b_i = next_state.cost;
        }

        for state in temp_pulled {
            if state.cost >= b_i {
                d.push(state);
                s_i.remove(&state.node_idx);
            }
        }
        if s_i.is_empty() && !d.is_empty() {
            let state = d.pop().unwrap();
            s_i.insert(state.node_idx);
        }

        if s_i.is_empty() {
            break;
        }

        let (b_i_prime, u_i) = bmssp(l - 1, b_i, s_i.clone(), graph, node_map, k, t, d_hat, pred);
        u_total.extend(u_i.iter());

        let mut k_set = Vec::new();

        for &u_node in &u_i {
            let u_name = node_map.iter().find(|(_, &v)| v == u_node).unwrap().0;
            if let Some(edges) = graph.get(u_name) {
                for edge in edges {
                    let v_idx = *node_map.get(&edge.node).unwrap();
                    if d_hat[u_node] != u32::MAX && d_hat[u_node] + edge.weight < d_hat[v_idx] {
                        d_hat[v_idx] = d_hat[u_node] + edge.weight;
                        pred[v_idx] = Some(u_node);

                        if d_hat[v_idx] >= b_i && d_hat[v_idx] < b {
                            d.push(State {
                                cost: d_hat[v_idx],
                                node_idx: v_idx,
                            });
                        } else if d_hat[v_idx] >= b_i_prime && d_hat[v_idx] < b_i {
                            k_set.push(State {
                                cost: d_hat[v_idx],
                                node_idx: v_idx,
                            });
                        }
                    }
                }
            }
        }
        for &x in &s_i {
            if d_hat[x] >= b_i_prime && d_hat[x] < b_i {
                k_set.push(State {
                    cost: d_hat[x],
                    node_idx: x,
                });
            }
        }
        for state in k_set {
            d.push(state);
        }
    }

    let final_b = d.peek().map_or(b, |s| s.cost);
    u_total.extend(w.iter().filter(|&&idx| d_hat[idx] < final_b));

    (final_b, u_total)
}

fn find_pivots(
    b: u32,
    s: &HashSet<usize>,
    graph: &Graph,
    node_map: &HashMap<String, usize>,
    k: usize,
    d_hat: &mut Vec<u32>,
    pred: &mut Vec<Option<usize>>,
) -> (HashSet<usize>, HashSet<usize>) {
    let mut w = s.clone();
    let mut w_curr = s.clone();

    for _ in 0..k {
        let mut w_next = HashSet::new();
        for &u_idx in &w_curr {
            let u_name = node_map.iter().find(|(_, &v)| v == u_idx).unwrap().0;
            if let Some(edges) = graph.get(u_name) {
                for edge in edges {
                    let v_idx = *node_map.get(&edge.node).unwrap();
                    if d_hat[u_idx] != u32::MAX && d_hat[u_idx] + edge.weight < d_hat[v_idx] {
                        if d_hat[u_idx] + edge.weight < b {
                            d_hat[v_idx] = d_hat[u_idx] + edge.weight;
                            pred[v_idx] = Some(u_idx);
                            w_next.insert(v_idx);
                        }
                    }
                }
            }
        }
        w.extend(w_next.iter());
        w_curr = w_next;
        if w.len() > k * s.len() {
            return (s.clone(), w);
        }
    }

    let mut tree_sizes = HashMap::new();
    let mut roots = HashMap::new();
    for &node in &w {
        roots.insert(node, node);
    }

    fn find_root(node: usize, roots: &mut HashMap<usize, usize>) -> usize {
        if roots[&node] == node {
            return node;
        }
        let root = find_root(roots[&node], roots);
        roots.insert(node, root);
        root
    }

    for &v_idx in &w {
        if let Some(Some(u_idx)) = pred.get(v_idx) {
            if w.contains(u_idx) {
                let root_u = find_root(*u_idx, &mut roots);
                let root_v = find_root(v_idx, &mut roots);
                if root_u != root_v {
                    roots.insert(root_v, root_u);
                }
            }
        }
    }

    for &node in &w {
        let root = find_root(node, &mut roots);
        *tree_sizes.entry(root).or_insert(0) += 1;
    }

    let pivots = s
        .iter()
        .filter(|&&node_idx| tree_sizes.get(&node_idx).unwrap_or(&0) >= &k)
        .cloned()
        .collect();
    (pivots, w)
}

fn base_case(
    b: u32,
    start_node: usize,
    graph: &Graph,
    node_map: &HashMap<String, usize>,
    k: usize,
    d_hat: &mut Vec<u32>,
    pred: &mut Vec<Option<usize>>,
) -> (u32, HashSet<usize>) {
    let mut u0 = HashSet::new();
    let mut heap = BinaryHeap::new();

    u0.insert(start_node);
    heap.push(State {
        cost: d_hat[start_node],
        node_idx: start_node,
    });

    let mut final_nodes = Vec::new();

    while let Some(state) = heap.pop() {
        if u0.len() >= k + 1 {
            final_nodes.push(state);
            break;
        }

        final_nodes.push(state);

        let u_name = node_map
            .iter()
            .find(|(_, &v)| v == state.node_idx)
            .unwrap()
            .0;
        if let Some(edges) = graph.get(u_name) {
            for edge in edges {
                let v_idx = *node_map.get(&edge.node).unwrap();
                if d_hat[state.node_idx] != u32::MAX
                    && d_hat[state.node_idx] + edge.weight < d_hat[v_idx]
                {
                    if d_hat[state.node_idx] + edge.weight < b {
                        d_hat[v_idx] = d_hat[state.node_idx] + edge.weight;
                        pred[v_idx] = Some(state.node_idx);
                        heap.push(State {
                            cost: d_hat[v_idx],
                            node_idx: v_idx,
                        });
                        u0.insert(v_idx);
                    }
                }
            }
        }
    }

    if u0.len() <= k {
        (b, u0)
    } else {
        final_nodes.sort_by_key(|s| s.cost);
        let new_b = final_nodes.last().unwrap().cost;
        let final_u = final_nodes
            .into_iter()
            .filter(|s| s.cost < new_b)
            .map(|s| s.node_idx)
            .collect();
        (new_b, final_u)
    }
}

fn build_node_maps(graph: &Graph) -> (HashMap<String, usize>, HashMap<usize, String>) {
    let mut node_map = HashMap::new();
    let mut rev_node_map = HashMap::new();
    let mut idx = 0;
    for node_name in graph.keys() {
        if !node_map.contains_key(node_name) {
            node_map.insert(node_name.clone(), idx);
            rev_node_map.insert(idx, node_name.clone());
            idx += 1;
        }
    }
    (node_map, rev_node_map)
}

fn reconstruct_path(
    start_idx: usize,
    end_idx: usize,
    prev: &[Option<usize>],
    rev_node_map: &HashMap<usize, String>,
) -> Vec<String> {
    let mut path = VecDeque::new();
    let mut current = Some(end_idx);
    while let Some(idx) = current {
        path.push_front(rev_node_map.get(&idx).unwrap().clone());
        if idx == start_idx {
            break;
        }
        current = prev[idx];
    }
    path.into()
}
