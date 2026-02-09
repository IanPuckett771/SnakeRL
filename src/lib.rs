use numpy::ndarray::{Array1, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray3};
use pyo3::prelude::*;
use std::collections::VecDeque;

/// Ray-casting directions: N, NE, E, SE, S, SW, W, NW
const RAY_DIRS: [(i32, i32); 8] = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
];

/// Cardinal directions: up, down, left, right
const CARDINAL_4: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

/// Encode game state into a 44-feature flat vector.
///
/// Features (44 total):
///   0-1:   Head position (normalized x, y)
///   2-3:   Food direction (normalized dx, dy)
///   4-7:   Direction one-hot
///   8-11:  1-step danger (cardinal)
///   12-15: Flood fill reachable (cardinal)
///   16-39: Ray-casting (8 dirs x 3: dist_wall, dist_body, dist_food)
///   40:    Snake length (normalized)
///   41-42: Tail direction (normalized)
///   43:    Body density ahead
#[pyfunction]
fn encode_state<'py>(
    py: Python<'py>,
    snake: Vec<(usize, usize)>,
    food: (usize, usize),
    walls: Vec<(usize, usize)>,
    direction: &str,
    width: usize,
    height: usize,
) -> Bound<'py, PyArray1<f32>> {
    let mut features = Array1::<f32>::zeros(44);
    let grid_size = height * width;
    let max_dim = width.max(height) as f32;

    let snake_len = snake.len();
    let (head_x, head_y) = snake[0];
    let (food_x, food_y) = food;
    let (tail_x, tail_y) = snake[snake_len - 1];

    let head_xi = head_x as i32;
    let head_yi = head_y as i32;
    let wi = width as i32;
    let hi = height as i32;

    // Build flat lookup arrays
    // blocked_collision: body (excl tail) + walls — for 1-step danger
    let mut blocked_collision = vec![false; grid_size];
    for &(sx, sy) in &snake[..snake_len - 1] {
        blocked_collision[sy * width + sx] = true;
    }
    for &(wx, wy) in &walls {
        blocked_collision[wy * width + wx] = true;
    }

    // blocked_flood: full body + walls — for flood fill
    let mut blocked_flood = vec![false; grid_size];
    for &(sx, sy) in &snake {
        blocked_flood[sy * width + sx] = true;
    }
    for &(wx, wy) in &walls {
        blocked_flood[wy * width + wx] = true;
    }

    // body_set: body (excl head) — for ray-casting
    let mut body_set = vec![false; grid_size];
    for &(sx, sy) in &snake[1..] {
        body_set[sy * width + sx] = true;
    }

    // wall_set: for ray-casting wall detection
    let mut wall_set = vec![false; grid_size];
    for &(wx, wy) in &walls {
        wall_set[wy * width + wx] = true;
    }

    // Direction index
    let dir_idx: usize = match direction {
        "up" => 0,
        "down" => 1,
        "left" => 2,
        "right" => 3,
        _ => 0,
    };

    // Head position (normalized)
    features[0] = head_x as f32 / width as f32;
    features[1] = head_y as f32 / height as f32;

    // Food direction (normalized)
    features[2] = (food_x as f32 - head_x as f32) / width as f32;
    features[3] = (food_y as f32 - head_y as f32) / height as f32;

    // Direction one-hot
    features[4 + dir_idx] = 1.0;

    // Danger + reachable space per cardinal direction
    for (i, &(dx, dy)) in CARDINAL_4.iter().enumerate() {
        let nx = head_xi + dx;
        let ny = head_yi + dy;

        let blocked_1 = nx < 0
            || nx >= wi
            || ny < 0
            || ny >= hi
            || blocked_collision[ny as usize * width + nx as usize];

        if blocked_1 {
            features[8 + i] = 1.0; // danger
            features[12 + i] = 0.0; // reachable = 0
        } else {
            // Flood fill (BFS, cap=100)
            let nxu = nx as usize;
            let nyu = ny as usize;
            let start_flat = nyu * width + nxu;

            if blocked_flood[start_flat] {
                features[12 + i] = 0.0;
            } else {
                let mut visited = vec![false; grid_size];
                visited[start_flat] = true;
                let mut queue = VecDeque::with_capacity(128);
                queue.push_back((nxu, nyu));
                let mut count: u32 = 0;
                let max_count: u32 = 100;

                while let Some((cx, cy)) = queue.pop_front() {
                    if count >= max_count {
                        break;
                    }
                    count += 1;
                    for &(ddx, ddy) in &CARDINAL_4 {
                        let nnx = cx as i32 + ddx;
                        let nny = cy as i32 + ddy;
                        if nnx >= 0 && nnx < wi && nny >= 0 && nny < hi {
                            let nnxu = nnx as usize;
                            let nnyu = nny as usize;
                            let flat = nnyu * width + nnxu;
                            if !visited[flat] && !blocked_flood[flat] {
                                visited[flat] = true;
                                queue.push_back((nnxu, nnyu));
                            }
                        }
                    }
                }
                features[12 + i] = count as f32 / 100.0;
            }
        }
    }

    // Ray-casting: 8 directions x 3 features (indices 16-39)
    for (r, &(rdx, rdy)) in RAY_DIRS.iter().enumerate() {
        let mut dist_wall: f32 = 1.0;
        let mut dist_body: f32 = 1.0;
        let mut dist_food: f32 = 1.0;
        let mut body_found = false;

        let mut cx = head_xi;
        let mut cy = head_yi;
        let mut step: u32 = 0;

        loop {
            cx += rdx;
            cy += rdy;
            step += 1;

            // Check boundary or wall
            if cx < 0 || cx >= wi || cy < 0 || cy >= hi {
                dist_wall = step as f32 / max_dim;
                break;
            }
            let flat = cy as usize * width + cx as usize;
            if wall_set[flat] {
                dist_wall = step as f32 / max_dim;
                break;
            }

            // Check body (first hit only)
            if !body_found && body_set[flat] {
                dist_body = step as f32 / max_dim;
                body_found = true;
            }

            // Check food
            if cx as usize == food_x && cy as usize == food_y {
                dist_food = step as f32 / max_dim;
            }
        }

        let base = 16 + r * 3;
        features[base] = dist_wall;
        features[base + 1] = dist_body;
        features[base + 2] = dist_food;
    }

    // Snake length normalized
    features[40] = snake_len as f32 / (width * height) as f32;

    // Tail direction (normalized)
    features[41] = (tail_x as f32 - head_x as f32) / width as f32;
    features[42] = (tail_y as f32 - head_y as f32) / height as f32;

    // Body density ahead
    let (move_dx, move_dy): (i32, i32) = match direction {
        "up" => (0, -1),
        "down" => (0, 1),
        "left" => (-1, 0),
        "right" => (1, 0),
        _ => (1, 0),
    };
    let mut body_ahead: u32 = 0;
    for &(bx, by) in &snake[1..] {
        let dot = (bx as i32 - head_xi) * move_dx + (by as i32 - head_yi) * move_dy;
        if dot > 0 {
            body_ahead += 1;
        }
    }
    features[43] = body_ahead as f32 / (snake_len as u32 - 1).max(1) as f32;

    features.into_pyarray_bound(py).into()
}

/// Encode game state as a 7-channel grid for CNN input.
///
/// Channels:
///   0: Snake head — 1.0 at head position
///   1: Snake body — decaying gradient from 1.0 (neck) to 0.0 (tail)
///   2: Snake tail — 1.0 at tail position
///   3: Food — 1.0 at food cell
///   4: Walls — 1.0 at each wall cell
///   5: Direction — gradient in movement direction (full-board context)
///   6: Reachability — BFS from head: 1.0 = reachable, 0.0 = blocked
#[pyfunction]
fn encode_state_grid<'py>(
    py: Python<'py>,
    snake: Vec<(usize, usize)>,
    food: (usize, usize),
    _food_points: i32,
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

    // Channel 3: Food
    let (food_x, food_y) = food;
    grid[[3, food_y, food_x]] = 1.0;

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
    let grid_size = height * width;
    let mut blocked = vec![false; grid_size];
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
    m.add_function(wrap_pyfunction!(encode_state, m)?)?;
    m.add_function(wrap_pyfunction!(encode_state_grid, m)?)?;
    Ok(())
}
