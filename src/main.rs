use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

const N: usize = 9;
const SIZE: usize = N * N;
type Mask = u16;

fn full_mask() -> Mask { (1 << N) - 1 }

#[derive(Clone)]
struct Board {
    cells: [Mask; SIZE],
}

impl Board {
    fn from_str(s: &str) -> Option<Self> {
        let mut cells = [0u16; SIZE];
        let ss: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
        if ss.len() != SIZE { return None; }
        for (i, ch) in ss.into_iter().enumerate() {
            if ch == '0' || ch == '.' { cells[i] = full_mask(); }
            else if ch.is_digit(10) {
                let v = ch.to_digit(10).unwrap() as usize;
                if v < 1 || v > N { return None; }
                cells[i] = 1 << (v - 1);
            } else { return None; }
        }
        let mut b = Board { cells };
        if !b.propagate_all() { return None; }
        Some(b)
    }

    fn is_solved(&self) -> bool {
        self.cells.iter().all(|&m| m.count_ones() == 1)
    }

    fn row(i: usize) -> usize { i / N }
    fn col(i: usize) -> usize { i % N }
    fn peers(idx: usize) -> Vec<usize> {
        let mut ps = Vec::with_capacity(20);
        let r = Self::row(idx);
        let c = Self::col(idx);
        for cc in 0..N { let j = r * N + cc; if j != idx { ps.push(j); } }
        for rr in 0..N { let j = rr * N + c; if j != idx && !ps.contains(&j) { ps.push(j); } }
        let br = (r / 3) * 3;
        let bc = (c / 3) * 3;
        for rr in br..br+3 {
            for cc in bc..bc+3 {
                let j = rr * N + cc;
                if j != idx && !ps.contains(&j) { ps.push(j); }
            }
        }
        ps
    }

    fn propagate_all(&mut self) -> bool {
        let mut queue: Vec<usize> = self.cells.iter().enumerate()
            .filter(|(_, &m)| m.count_ones() == 1)
            .map(|(i, _)| i)
            .collect();

        while let Some(idx) = queue.pop() {
            let val_mask = self.cells[idx];
            if val_mask == 0 { return false; }
            for peer in Board::peers(idx) {
                let before = self.cells[peer];
                if before & val_mask != 0 {
                    let now = before & (!val_mask);
                    if now == 0 { return false; }
                    if now != before {
                        self.cells[peer] = now;
                        if now.count_ones() == 1 { queue.push(peer); }
                    }
                }
            }
        }
        true
    }

    fn assign_and_propagate(&self, idx: usize, val_mask: Mask) -> Option<Board> {
        let mut nb = self.clone();
        nb.cells[idx] = val_mask;
        if nb.propagate_all() { Some(nb) } else { None }
    }

    fn choose_min_candidate(&self) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for i in 0..SIZE {
            let c = self.cells[i].count_ones() as usize;
            if c == 0 { return None; }
            if c > 1 {
                match best {
                    None => best = Some((i, c)),
                    Some((_, bc)) if c < bc => best = Some((i, c)),
                    _ => {}
                }
            }
        }
        best
    }

    fn candidates_masks(mask: Mask) -> Vec<Mask> {
        let mut v = Vec::new();
        for bit in 0..N {
            if (mask & (1 << bit)) != 0 { v.push(1 << bit); }
        }
        v
    }

    fn to_pretty(&self) -> String {
        let mut s = String::new();
        for r in 0..N {
            for c in 0..N {
                let i = r * N + c;
                if self.cells[i].count_ones() == 1 {
                    let val = self.cells[i].trailing_zeros() + 1;
                    s.push_str(&format!("{} ", val));
                } else { s.push_str(". "); }
            }
            s.push('\n');
        }
        s
    }
    fn to_grid(&self) -> String {
        let mut s = String::new();
        for r in 0..N {
            if r % 3 == 0 && r != 0 {
                s.push_str("------+-------+------\n");
            }
            for c in 0..N {
                if c % 3 == 0 && c != 0 {
                    s.push_str("| ");
                }
                let i = r * N + c;
                if self.cells[i].count_ones() == 1 {
                    let val = self.cells[i].trailing_zeros() + 1;
                    s.push_str(&format!("{} ", val));
                } else {
                    s.push_str(". ");
                }
            }
            s.push('\n');
        }
        s
    }

}

// Backtracking secuencial
fn solve_backtracking(board: &Board, stop_flag: &AtomicBool) -> Option<Board> {
    if stop_flag.load(Ordering::SeqCst) { return None; }
    if board.is_solved() { return Some(board.clone()); }
    if let Some((idx, _)) = board.choose_min_candidate() {
        for cand in Board::candidates_masks(board.cells[idx]) {
            if let Some(child) = board.assign_and_propagate(idx, cand) {
                if let Some(sol) = solve_backtracking(&child, stop_flag) {
                    return Some(sol);
                }
            }
        }
    }
    None
}

// Solver paralelo dividiendo trabajo inicial
fn solve_parallel_split(board: &Board, k_threads: usize) -> Option<Board> {
    let stop_flag = Arc::new(AtomicBool::new(false));
    let solution = Arc::new(Mutex::new(None::<Board>));

    // Elegir primera celda mínima para ramificación inicial
    if let Some((idx, _)) = board.choose_min_candidate() {
        let candidates = Board::candidates_masks(board.cells[idx]);
        let mut handles = Vec::new();

        for (i, cand) in candidates.into_iter().enumerate() {
            if i >= k_threads { break; } // limitar número de hilos
            let mut b = board.assign_and_propagate(idx, cand).unwrap();
            let stop_c = stop_flag.clone();
            let sol_c = solution.clone();
            let handle = thread::spawn(move || {
                if let Some(sol) = solve_backtracking(&b, &stop_c) {
                    let mut guard = sol_c.lock().unwrap();
                    *guard = Some(sol.clone());
                    stop_c.store(true, Ordering::SeqCst);
                }
            });
            handles.push(handle);
        }

        for h in handles { let _ = h.join(); }

        let guard = solution.lock().unwrap();
        return guard.clone();
    }
    None
}

fn main() {
    let puzzle = "\
        800000000\
        003600000\
        070090200\
        050007000\
        000045700\
        000100030\
        001000068\
        008500010\
        090000400\
        ";

    let board = Board::from_str(puzzle).expect("Invalid board");
    let max_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) + 1;

    println!("Benchmark Sudoku Solver (1..= {} threads)", max_threads);

    let mut solution_board: Option<Board> = None;
    let mut results: Vec<(usize, f64)> = Vec::new(); // (k, tiempo en ms)

    for k in 1..=max_threads {
        let start = Instant::now();
        let sol = solve_parallel_split(&board, k);
        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

        if sol.is_some() && solution_board.is_none() {
            solution_board = sol.clone();
        }

        println!("Threads {:>2}: solved in {:.4} ms", k, elapsed_ms);
        results.push((k, elapsed_ms));
    }

    // --- Calcular Speedup y Efficiency ---
    if let Some((_, t1)) = results.first() {
        println!("\nSummary (avg times):");
        println!("k\tT_k(ms)\tSpeedup\tEfficiency");
        for (k, tk) in &results {
            let speedup = t1 / tk;
            let efficiency = speedup / (*k as f64);
            println!("{}\t{:.4}\t{:.4}\t{:.4}", k, tk, speedup, efficiency);
        }
    }

    if let Some(board) = solution_board {
        println!("\nSudoku resuelto:\n{}", board.to_grid());
    } else {
        println!("No se encontró solución.");
    }
}
