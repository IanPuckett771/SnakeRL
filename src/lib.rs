use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use std::collections::VecDeque;

/// Encode game state as a 7-channel grid for CNN input.
///
/// Channels:
///   0: Snake head — 1.0 at head position
///   1: Snake body — decaying gradient from 1.0 (neck) to 0.0 (tail)
///   2: Snake tail — 1.0 at tail position
///   3: Food — food_points / 20.0 at food cell
///   4: Walls — 1.0 at each wall cell
///   5: Direction — gradient in movement direction (full-board context)
///   6: Reachability — BFS from head: 1.0 = reachable, 0.0 = blocked
#[pyfunction]
fn encode_state_grid<'py>(
    py: Python<'py>,
    snake: Vec<(usize, usize)>,
    food: (usize, usize),
    food_points: i32,
    walls: Vec<(usize, usize)>,
    direction: &str,
    width: usize,
    height: usize,
) -> Bound<'py, PyArray3<f32>> {
    let mut grid = Array3::<f32>::zeros((7, height, width));

    let snake_len = snake.len();
    let (head_x, head_y) = snake[0];

    // Channel 0: Head
    grid[[0, head_y, head_x]] = 1.0;

    // Channel 1: Body gradient (1.0 at neck → 0.0 at tail)
    if snake_len > 2 {
        for (i, &(bx, by)) in snake[1..snake_len - 1].iter().enumerate() {
            let idx = i + 1; // 1-based index within full snake
            grid[[1, by, bx]] = 1.0 - (idx as f32 / (snake_len - 1) as f32);
        }
    }

    // Channel 2: Tail
    let (tail_x, tail_y) = snake[snake_len - 1];
    grid[[2, tail_y, tail_x]] = 1.0;

    // Channel 3: Food (scaled by points)
    let (food_x, food_y) = food;
    grid[[3, food_y, food_x]] = food_points as f32 / 20.0;

    // Channel 4: Walls
    for &(wx, wy) in &walls {
        grid[[4, wy, wx]] = 1.0;
    }

    // Channel 5: Direction gradient
    let (dx, dy): (i32, i32) = match direction {
        "up" => (0, -1),
        "down" => (0, 1),
        "left" => (-1, 0),
        "right" => (1, 0),
        _ => (1, 0),
    };

    let head_x_i = head_x as i32;
    let head_y_i = head_y as i32;

    if dx != 0 {
        // Horizontal gradient
        let denom = (width as i32 - 1).max(1) as f32;
        for col in 0..width {
            let val = (col as i32 - head_x_i) * dx;
            let normalized = val as f32 / denom;
            for row in 0..height {
                grid[[5, row, col]] = normalized;
            }
        }
    } else {
        // Vertical gradient
        let denom = (height as i32 - 1).max(1) as f32;
        for row in 0..height {
            let val = (row as i32 - head_y_i) * dy;
            let normalized = val as f32 / denom;
            for col in 0..width {
                grid[[5, row, col]] = normalized;
            }
        }
    }

    // Channel 6: Reachability via BFS from head
    // Build blocked set as flat bool array for O(1) lookup
    let grid_size = height * width;
    let mut blocked = vec![false; grid_size];
    // Body (excluding head) blocks movement
    for &(sx, sy) in &snake[1..] {
        blocked[sy * width + sx] = true;
    }
    for &(wx, wy) in &walls {
        blocked[wy * width + wx] = true;
    }

    let mut visited = vec![false; grid_size];
    visited[head_y * width + head_x] = true;
    grid[[6, head_y, head_x]] = 1.0;

    let mut queue = VecDeque::with_capacity(grid_size);
    queue.push_back((head_x, head_y));

    let directions: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

    while let Some((cx, cy)) = queue.pop_front() {
        for &(ddx, ddy) in &directions {
            let nx = cx as i32 + ddx;
            let ny = cy as i32 + ddy;
            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                let nxu = nx as usize;
                let nyu = ny as usize;
                let flat = nyu * width + nxu;
                if !visited[flat] && !blocked[flat] {
                    visited[flat] = true;
                    grid[[6, nyu, nxu]] = 1.0;
                    queue.push_back((nxu, nyu));
                }
            }
        }
    }

    grid.into_pyarray_bound(py).into()
}

#[pymodule]
fn snakerl_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_state_grid, m)?)?;
    Ok(())
}
