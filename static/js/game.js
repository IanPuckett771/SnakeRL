/**
 * Game Controller for SnakeRL
 * Handles game rendering, input, and state management
 */

class GameController {
    constructor() {
        // Canvas and rendering
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.cellSize = 20;

        // Game state
        this.gameActive = false;
        this.currentScore = 0;
        this.currentMode = 'play';
        this.snakeColor = '#00ff00';
        this.boardWidth = 20;
        this.boardHeight = 20;

        // UI Elements
        this.scoreDisplay = document.getElementById('currentScore');
        this.finalScoreDisplay = document.getElementById('finalScore');
        this.gameOverOverlay = document.getElementById('gameOverOverlay');
        this.leaderboardList = document.getElementById('leaderboardList');
        this.checkpointGroup = document.getElementById('checkpointGroup');
        this.checkpointSelect = document.getElementById('checkpoint');
        this.checkpointInfo = document.getElementById('checkpointInfo');
        this.checkpointName = document.getElementById('checkpointName');
        this.checkpointMeta = document.getElementById('checkpointMeta');
        this.trainingProgress = document.getElementById('trainingProgress');
        this.progressBar = document.getElementById('progressBar');
        this.trainingStats = document.getElementById('trainingStats');
        
        // Track last checkpoint used for "Play Again"
        this.lastCheckpoint = null;
        this.lastMode = 'play';
        
        // Check for training progress periodically
        this.trainingCheckInterval = null;
        this.checkpointRefreshInterval = null;

        // Settings inputs
        this.boardWidthInput = document.getElementById('boardWidth');
        this.boardHeightInput = document.getElementById('boardHeight');
        this.snakeColorInput = document.getElementById('snakeColor');
        this.gameSpeedSelect = document.getElementById('gameSpeed');

        // Buttons
        this.startBtn = document.getElementById('startBtn');
        this.playModeBtn = document.getElementById('playModeBtn');
        this.watchModeBtn = document.getElementById('watchModeBtn');
        this.submitScoreBtn = document.getElementById('submitScoreBtn');
        this.playAgainBtn = document.getElementById('playAgainBtn');
        this.closeOverlayBtn = document.getElementById('closeOverlayBtn');
        this.playerNameInput = document.getElementById('playerName');

        this.init();
    }

    /**
     * Initialize the game controller
     */
    init() {
        this.setupEventListeners();
        this.setupWebSocket();
        this.fetchLeaderboard();
        this.drawEmptyGrid();
        this.startTrainingProgressCheck();
        this.startCheckpointRefresh();
    }
    
    /**
     * Start checking for training progress
     */
    startTrainingProgressCheck() {
        // Check every 2 seconds if training is happening
        this.trainingCheckInterval = setInterval(() => {
            this.checkTrainingProgress();
        }, 2000);
    }
    
    /**
     * Start auto-refreshing checkpoints when in agent mode
     */
    startCheckpointRefresh() {
        // Refresh checkpoints every 5 seconds if in agent mode
        this.checkpointRefreshInterval = setInterval(() => {
            if (this.currentMode === 'agent') {
                this.fetchCheckpoints();
            }
        }, 5000); // Check every 5 seconds
    }
    
    /**
     * Check if training is in progress by looking for checkpoint file updates
     */
    async checkTrainingProgress() {
        try {
            const response = await fetch('/training-status');
            if (response.ok) {
                const data = await response.json();
                if (data.training) {
                    this.trainingProgress.classList.add('visible');
                    const progress = Math.min(100, (data.elapsed / data.duration) * 100);
                    this.progressBar.style.width = `${progress}%`;
                    this.progressBar.textContent = `${Math.round(progress)}%`;
                    
                    let statsText = '';
                    if (data.episodes) {
                        statsText += `Episodes: ${data.episodes}`;
                    }
                    if (data.avg_score !== undefined) {
                        if (statsText) statsText += ' • ';
                        statsText += `Avg Score: ${data.avg_score.toFixed(1)}`;
                    }
                    if (statsText) {
                        this.trainingStats.textContent = statsText;
                    }
                } else {
                    this.trainingProgress.classList.remove('visible');
                    // Refresh checkpoints if training just finished
                    if (this.currentMode === 'agent') {
                        this.fetchCheckpoints();
                    }
                }
            }
        } catch (error) {
            // Silently fail - training status is optional
        }
    }

    /**
     * Set up all event listeners
     */
    setupEventListeners() {
        // Keyboard controls
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));

        // Mode toggle buttons
        this.playModeBtn.addEventListener('click', () => this.setMode('play'));
        this.watchModeBtn.addEventListener('click', () => this.setMode('agent'));

        // Start button
        this.startBtn.addEventListener('click', () => this.startGame());

        // Game over overlay buttons
        this.submitScoreBtn.addEventListener('click', () => this.submitScore());
        this.playAgainBtn.addEventListener('click', () => this.playAgain());
        this.closeOverlayBtn.addEventListener('click', () => this.closeOverlay());

        // Allow Enter key to submit score
        this.playerNameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.submitScore();
            }
        });

        // Update snake color preview
        this.snakeColorInput.addEventListener('input', (e) => {
            this.snakeColor = e.target.value;
        });
    }

    /**
     * Set up WebSocket connection and message handling
     */
    setupWebSocket() {
        wsManager.onMessage((data) => this.handleMessage(data));

        // Connect to WebSocket
        wsManager.connect().catch(error => {
            console.error('Failed to connect to WebSocket:', error);
        });
    }

    /**
     * Handle incoming WebSocket messages
     * @param {Object} data - Parsed message data
     */
    handleMessage(data) {
        switch (data.type) {
            case 'state_update':
                // Game is active once we receive state updates
                if (!this.gameActive) {
                    this.gameActive = true;
                    this.currentScore = 0;
                    this.updateScoreDisplay(0);
                }
                this.renderState(data.state);
                break;
            case 'game_over':
                this.handleGameOver(data.final_score);
                break;
            case 'error':
                console.error('Server error:', data.message);
                alert(`Error: ${data.message}`);
                break;
        }
    }

    /**
     * Handle keyboard input
     * @param {KeyboardEvent} e - Keyboard event
     */
    handleKeyDown(e) {
        // Only handle game controls if in play mode and game is active
        if (this.currentMode !== 'play' || !this.gameActive) {
            return;
        }

        let direction = null;

        // Arrow keys
        switch (e.key) {
            case 'ArrowUp':
                direction = 'up';
                break;
            case 'ArrowDown':
                direction = 'down';
                break;
            case 'ArrowLeft':
                direction = 'left';
                break;
            case 'ArrowRight':
                direction = 'right';
                break;
            // WASD keys
            case 'w':
            case 'W':
                direction = 'up';
                break;
            case 's':
            case 'S':
                direction = 'down';
                break;
            case 'a':
            case 'A':
                direction = 'left';
                break;
            case 'd':
            case 'D':
                direction = 'right';
                break;
        }

        if (direction) {
            // Prevent default to stop page scrolling
            e.preventDefault();

            // Send action immediately for responsive feel
            wsManager.send({
                type: 'action',
                action: direction
            });
        }
    }

    /**
     * Set the game mode
     * @param {string} mode - 'play' or 'agent'
     */
    setMode(mode) {
        // Preserve checkpoint selection when switching modes
        const currentCheckpoint = this.checkpointSelect.value;
        
        this.currentMode = mode;

        // Update button states
        this.playModeBtn.classList.toggle('active', mode === 'play');
        this.watchModeBtn.classList.toggle('active', mode === 'agent');

        // Show/hide checkpoint dropdown
        this.checkpointGroup.classList.toggle('visible', mode === 'agent');

        // Fetch checkpoints if switching to agent mode (will preserve selection)
        if (mode === 'agent') {
            this.fetchCheckpoints();
        }

        // Update start button text
        this.startBtn.textContent = mode === 'play' ? 'Start Game' : 'Watch Agent';
    }

    /**
     * Fetch available checkpoints from the server
     */
    async fetchCheckpoints() {
        try {
            // Preserve current selection before repopulating
            const currentSelection = this.checkpointSelect.value;
            
            const response = await fetch('/checkpoints');
            if (!response.ok) {
                throw new Error('Failed to fetch checkpoints');
            }

            const data = await response.json();
            // Backend returns {checkpoints: [...]}, so extract the array
            const checkpoints = data.checkpoints || data;
            this.populateCheckpointDropdown(checkpoints);
            
            // Restore selection if it still exists
            if (currentSelection && this.checkpointSelect.querySelector(`option[value="${currentSelection}"]`)) {
                this.checkpointSelect.value = currentSelection;
            }
        } catch (error) {
            console.error('Error fetching checkpoints:', error);
            this.checkpointSelect.innerHTML = '<option value="">No checkpoints available</option>';
        }
    }

    /**
     * Populate the checkpoint dropdown with options
     * @param {Array} checkpoints - Array of checkpoint objects
     */
    populateCheckpointDropdown(checkpoints) {
        this.checkpointSelect.innerHTML = '<option value="">Default Agent (Heuristic)</option>';

        if (Array.isArray(checkpoints) && checkpoints.length > 0) {
            // Group checkpoints by algorithm and run, then sort stages
            const grouped = {};
            const timestamps = [];
            
            // First pass: collect all checkpoints and timestamps
            checkpoints.forEach(checkpoint => {
                const checkpointValue = typeof checkpoint === 'string' 
                    ? checkpoint 
                    : (checkpoint.path || checkpoint.name || checkpoint);
                
                // Extract algorithm name, timestamp, and stage
                // Pattern: algo_agent_20250125_123456_stage01.pt or algo_agent_stage01.pt (old format)
                const timestampMatch = checkpointValue.match(/^(.+?)_agent_(\d{8}_\d{6})(_stage(\d+))?\.pt$/);
                const oldMatch = checkpointValue.match(/^(.+?)_agent(_stage(\d+))?\.pt$/);
                
                if (timestampMatch) {
                    // New format with timestamp
                    const algoName = timestampMatch[1];
                    const timestamp = timestampMatch[2];
                    const stageNum = timestampMatch[4] ? parseInt(timestampMatch[4]) : 999;
                    
                    // Collect timestamp for finding newest
                    if (!timestamps.includes(timestamp)) {
                        timestamps.push(timestamp);
                    }
                    
                    const key = `${algoName}_${timestamp}`;
                    if (!grouped[key]) {
                        grouped[key] = { algoName, timestamp, items: [] };
                    }
                    grouped[key].items.push({ value: checkpointValue, stage: stageNum });
                } else if (oldMatch) {
                    // Old format without timestamp
                    const algoName = oldMatch[1];
                    const stageNum = oldMatch[3] ? parseInt(oldMatch[3]) : 999;
                    
                    const key = `${algoName}_old`;
                    if (!grouped[key]) {
                        grouped[key] = { algoName, timestamp: 'old', items: [] };
                    }
                    grouped[key].items.push({ value: checkpointValue, stage: stageNum });
                } else {
                    // Handle non-standard checkpoint names
                    if (!grouped['other_old']) {
                        grouped['other_old'] = { algoName: 'other', timestamp: 'old', items: [] };
                    }
                    grouped['other_old'].items.push({ value: checkpointValue, stage: 999 });
                }
            });
            
            // Find the most recent timestamp (newest training run)
            const newestTimestamp = timestamps.length > 0 
                ? timestamps.sort().reverse()[0]  // Sort descending, take first
                : null;
            
            // Sort by timestamp (newest first) and algorithm name, then by stage
            Object.keys(grouped).sort((a, b) => {
                const groupA = grouped[a];
                const groupB = grouped[b];
                // Sort by timestamp (newer first), then by algorithm name
                if (groupA.timestamp !== groupB.timestamp) {
                    if (groupA.timestamp === 'old') return 1;
                    if (groupB.timestamp === 'old') return -1;
                    return groupB.timestamp.localeCompare(groupA.timestamp); // Newest first
                }
                return groupA.algoName.localeCompare(groupB.algoName);
            }).forEach(key => {
                const group = grouped[key];
                const items = group.items;
                items.sort((a, b) => a.stage - b.stage);
                
                // Determine if this is the newest training run
                const isNewest = group.timestamp !== 'old' && group.timestamp === newestTimestamp;
                
                // Format timestamp for display
                let runLabel = '';
                if (isNewest) {
                    // Parse timestamp: 20250125_123456 -> Jan 25, 12:34:56
                    const ts = group.timestamp;
                    const date = ts.substring(0, 8);
                    const time = ts.substring(9);
                    const month = date.substring(4, 6);
                    const day = date.substring(6, 8);
                    const hour = time.substring(0, 2);
                    const min = time.substring(2, 4);
                    runLabel = ` [NEW - ${month}/${day} ${hour}:${min}]`;
                } else if (group.timestamp !== 'old') {
                    // Show timestamp for older runs
                    const ts = group.timestamp;
                    const date = ts.substring(0, 8);
                    const time = ts.substring(9);
                    const month = date.substring(4, 6);
                    const day = date.substring(6, 8);
                    const hour = time.substring(0, 2);
                    const min = time.substring(2, 4);
                    runLabel = ` (${month}/${day} ${hour}:${min})`;
                } else {
                    runLabel = ' (Previous)';
                }
                
                items.forEach(item => {
                    const checkpointValue = item.value;
                    let checkpointLabel = '';
                    
                    // Format label nicely
                    if (item.stage < 999) {
                        checkpointLabel = `${group.algoName.toUpperCase()} - Stage ${item.stage}/10${runLabel}`;
                    } else {
                        checkpointLabel = `${group.algoName.toUpperCase()} (final)${runLabel}`;
                    }
                    
                    const option = document.createElement('option');
                    option.value = checkpointValue;
                    option.textContent = checkpointLabel;
                    // Highlight newest checkpoints only
                    if (isNewest) {
                        option.style.fontWeight = 'bold';
                        option.style.color = '#00ff00';
                    }
                    this.checkpointSelect.appendChild(option);
                });
            });
        }
        // If no checkpoints, default option is already set above
    }

    /**
     * Start a new game
     */
    startGame() {
        if (!wsManager.isConnected()) {
            alert('Not connected to server. Please wait...');
            wsManager.connect();
            return;
        }

        // Get settings
        const width = parseInt(this.boardWidthInput.value) || 20;
        const height = parseInt(this.boardHeightInput.value) || 20;
        this.snakeColor = this.snakeColorInput.value;

        // Validate dimensions
        const clampedWidth = Math.min(50, Math.max(10, width));
        const clampedHeight = Math.min(50, Math.max(10, height));

        // Store for leaderboard submission
        this.boardWidth = clampedWidth;
        this.boardHeight = clampedHeight;

        // Update canvas size
        this.resizeCanvas(clampedWidth, clampedHeight);

        // Get speed setting
        const speed = parseFloat(this.gameSpeedSelect.value) || 0.15;

        // Build config
        const config = {
            type: 'start_game',
            width: clampedWidth,
            height: clampedHeight,
            mode: this.currentMode,
            snake_color: this.snakeColor,
            speed: speed
        };

        // Add checkpoint if in agent mode (optional - can use default agent)
        if (this.currentMode === 'agent') {
            const checkpoint = this.checkpointSelect.value;
            if (checkpoint) {
                config.checkpoint = checkpoint;
                this.lastCheckpoint = checkpoint;
            } else {
                this.lastCheckpoint = null;
            }
            this.lastMode = 'agent';
        } else {
            this.lastMode = 'play';
        }

        // Hide game over overlay if visible
        this.gameOverOverlay.classList.remove('visible');

        // Send start game message
        wsManager.send(config);

        // Reset score display
        this.currentScore = 0;
        this.updateScoreDisplay(0);
        
        // Update checkpoint display
        this.updateCheckpointDisplay(config.checkpoint);
    }
    
    /**
     * Update checkpoint information display
     * @param {string} checkpoint - Checkpoint name or null
     */
    updateCheckpointDisplay(checkpoint) {
        if (this.currentMode === 'agent') {
            this.checkpointInfo.style.display = 'block';
            if (checkpoint) {
                this.checkpointName.textContent = checkpoint;
                // Fetch checkpoint metadata if available
                this.fetchCheckpointMetadata(checkpoint);
            } else {
                this.checkpointName.textContent = 'Default Agent (Heuristic)';
                this.checkpointMeta.textContent = 'Using rule-based strategy';
            }
        } else {
            this.checkpointInfo.style.display = 'none';
        }
    }
    
    /**
     * Fetch checkpoint metadata from server
     * @param {string} checkpoint - Checkpoint filename
     */
    async fetchCheckpointMetadata(checkpoint) {
        try {
            const response = await fetch(`/checkpoint-info/${encodeURIComponent(checkpoint)}`);
            if (response.ok) {
                const data = await response.json();
                let metaText = '';
                if (data.episodes) {
                    metaText += `Trained for ${data.episodes} episodes`;
                }
                if (data.avg_score !== undefined) {
                    if (metaText) metaText += ' • ';
                    metaText += `Avg Score: ${data.avg_score.toFixed(1)}`;
                }
                if (data.epsilon !== undefined) {
                    if (metaText) metaText += ' • ';
                    metaText += `Epsilon: ${data.epsilon.toFixed(3)}`;
                }
                this.checkpointMeta.textContent = metaText || 'Trained agent';
            } else {
                this.checkpointMeta.textContent = 'Trained agent';
            }
        } catch (error) {
            console.error('Error fetching checkpoint metadata:', error);
            this.checkpointMeta.textContent = 'Trained agent';
        }
    }

    /**
     * Resize the canvas based on board dimensions
     * @param {number} width - Board width in cells
     * @param {number} height - Board height in cells
     */
    resizeCanvas(width, height) {
        // Calculate cell size to fit reasonably on screen
        const maxCanvasSize = 600;
        const maxDimension = Math.max(width, height);
        this.cellSize = Math.floor(maxCanvasSize / maxDimension);
        this.cellSize = Math.max(10, Math.min(30, this.cellSize)); // Clamp between 10 and 30

        this.canvas.width = width * this.cellSize;
        this.canvas.height = height * this.cellSize;
    }

    /**
     * Render the game state
     * @param {Object} state - Game state from server
     */
    renderState(state) {
        if (!state) return;

        const { width, height, snake, food, food_type, walls, score } = state;

        // Clear canvas
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw grid lines
        this.drawGrid(width, height);

        // Draw walls
        if (walls && walls.length > 0) {
            this.drawWalls(walls);
        }

        // Draw food
        if (food) {
            this.drawFood(food, food_type);
        }

        // Draw snake
        if (snake && snake.length > 0) {
            this.drawSnake(snake);
        }

        // Update score
        if (score !== undefined) {
            this.currentScore = score;
            this.updateScoreDisplay(score);
        }
    }

    /**
     * Draw the grid lines
     * @param {number} width - Grid width in cells
     * @param {number} height - Grid height in cells
     */
    drawGrid(width, height) {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        this.ctx.lineWidth = 1;

        // Vertical lines
        for (let x = 0; x <= width; x++) {
            this.ctx.beginPath();
            this.ctx.moveTo(x * this.cellSize, 0);
            this.ctx.lineTo(x * this.cellSize, height * this.cellSize);
            this.ctx.stroke();
        }

        // Horizontal lines
        for (let y = 0; y <= height; y++) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y * this.cellSize);
            this.ctx.lineTo(width * this.cellSize, y * this.cellSize);
            this.ctx.stroke();
        }
    }

    /**
     * Draw an empty grid (before game starts)
     */
    drawEmptyGrid() {
        const width = parseInt(this.boardWidthInput.value) || 20;
        const height = parseInt(this.boardHeightInput.value) || 20;

        this.resizeCanvas(width, height);

        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.drawGrid(width, height);
    }

    /**
     * Draw the snake
     * @param {Array} snake - Array of {x, y} positions
     */
    drawSnake(snake) {
        snake.forEach((segment, index) => {
            const x = segment.x * this.cellSize;
            const y = segment.y * this.cellSize;

            // Head is brighter
            if (index === 0) {
                this.ctx.fillStyle = this.snakeColor;
                this.ctx.shadowColor = this.snakeColor;
                this.ctx.shadowBlur = 10;
            } else {
                // Body segments are slightly darker
                this.ctx.fillStyle = this.adjustColorBrightness(this.snakeColor, -30);
                this.ctx.shadowBlur = 0;
            }

            // Draw rounded rectangle for each segment
            this.roundRect(x + 1, y + 1, this.cellSize - 2, this.cellSize - 2, 4);

            // Reset shadow
            this.ctx.shadowBlur = 0;
        });

        // Draw eyes on the head
        if (snake.length > 0) {
            this.drawEyes(snake[0], snake.length > 1 ? snake[1] : null);
        }
    }

    /**
     * Draw eyes on the snake head
     * @param {Object} head - Head position {x, y}
     * @param {Object} neck - Second segment position for direction
     */
    drawEyes(head, neck) {
        const x = head.x * this.cellSize;
        const y = head.y * this.cellSize;
        const eyeSize = this.cellSize / 6;

        // Determine direction for eye placement
        let offsetX = 0.3;
        let offsetY = 0.3;

        if (neck) {
            if (neck.x < head.x) { // Moving right
                offsetX = 0.6;
            } else if (neck.x > head.x) { // Moving left
                offsetX = 0.2;
            }
            if (neck.y < head.y) { // Moving down
                offsetY = 0.6;
            } else if (neck.y > head.y) { // Moving up
                offsetY = 0.2;
            }
        }

        this.ctx.fillStyle = '#fff';

        // Left eye
        this.ctx.beginPath();
        this.ctx.arc(
            x + this.cellSize * 0.35,
            y + this.cellSize * offsetY,
            eyeSize,
            0,
            Math.PI * 2
        );
        this.ctx.fill();

        // Right eye
        this.ctx.beginPath();
        this.ctx.arc(
            x + this.cellSize * 0.65,
            y + this.cellSize * offsetY,
            eyeSize,
            0,
            Math.PI * 2
        );
        this.ctx.fill();

        // Pupils
        this.ctx.fillStyle = '#000';
        this.ctx.beginPath();
        this.ctx.arc(
            x + this.cellSize * 0.35,
            y + this.cellSize * offsetY,
            eyeSize / 2,
            0,
            Math.PI * 2
        );
        this.ctx.fill();

        this.ctx.beginPath();
        this.ctx.arc(
            x + this.cellSize * 0.65,
            y + this.cellSize * offsetY,
            eyeSize / 2,
            0,
            Math.PI * 2
        );
        this.ctx.fill();
    }

    /**
     * Draw the food with color based on type
     * @param {Object} food - Food position {x, y}
     * @param {string} foodType - Food type/color: "red", "orange", "yellow", "green", "blue"
     */
    drawFood(food, foodType = 'red') {
        const x = food.x * this.cellSize;
        const y = food.y * this.cellSize;
        const centerX = x + this.cellSize / 2;
        const centerY = y + this.cellSize / 2;
        const radius = (this.cellSize - 4) / 2;

        // Color mapping for different treat types
        const colorMap = {
            'red': {
                shadow: '#ff0000',
                light: '#ff6666',
                mid: '#ff0000',
                dark: '#cc0000'
            },
            'orange': {
                shadow: '#ff8800',
                light: '#ffaa44',
                mid: '#ff8800',
                dark: '#cc6600'
            },
            'yellow': {
                shadow: '#ffdd00',
                light: '#ffee66',
                mid: '#ffdd00',
                dark: '#ccaa00'
            },
            'green': {
                shadow: '#00ff00',
                light: '#66ff66',
                mid: '#00ff00',
                dark: '#00cc00'
            },
            'blue': {
                shadow: '#0088ff',
                light: '#44aaff',
                mid: '#0088ff',
                dark: '#0066cc'
            }
        };

        const colors = colorMap[foodType] || colorMap['red'];

        // Draw glowing food
        this.ctx.shadowColor = colors.shadow;
        this.ctx.shadowBlur = 15;

        // Gradient for 3D effect
        const gradient = this.ctx.createRadialGradient(
            centerX - radius / 3,
            centerY - radius / 3,
            radius / 4,
            centerX,
            centerY,
            radius
        );
        gradient.addColorStop(0, colors.light);
        gradient.addColorStop(0.5, colors.mid);
        gradient.addColorStop(1, colors.dark);

        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        this.ctx.fill();

        // Reset shadow
        this.ctx.shadowBlur = 0;
    }

    /**
     * Draw walls
     * @param {Array} walls - Array of {x, y} wall positions
     */
    drawWalls(walls) {
        walls.forEach(wall => {
            const x = wall.x * this.cellSize;
            const y = wall.y * this.cellSize;

            // Draw wall with dark gray color
            this.ctx.fillStyle = '#333333';
            this.ctx.strokeStyle = '#555555';
            this.ctx.lineWidth = 2;

            // Draw filled rectangle with border
            this.ctx.fillRect(x + 1, y + 1, this.cellSize - 2, this.cellSize - 2);
            this.ctx.strokeRect(x + 1, y + 1, this.cellSize - 2, this.cellSize - 2);

            // Add some texture with darker lines
            this.ctx.strokeStyle = '#222222';
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.moveTo(x + 2, y + 2);
            this.ctx.lineTo(x + this.cellSize - 2, y + this.cellSize - 2);
            this.ctx.moveTo(x + this.cellSize - 2, y + 2);
            this.ctx.lineTo(x + 2, y + this.cellSize - 2);
            this.ctx.stroke();
        });
    }

    /**
     * Draw a rounded rectangle
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} width - Width
     * @param {number} height - Height
     * @param {number} radius - Corner radius
     */
    roundRect(x, y, width, height, radius) {
        this.ctx.beginPath();
        this.ctx.moveTo(x + radius, y);
        this.ctx.lineTo(x + width - radius, y);
        this.ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        this.ctx.lineTo(x + width, y + height - radius);
        this.ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        this.ctx.lineTo(x + radius, y + height);
        this.ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        this.ctx.lineTo(x, y + radius);
        this.ctx.quadraticCurveTo(x, y, x + radius, y);
        this.ctx.closePath();
        this.ctx.fill();
    }

    /**
     * Adjust color brightness
     * @param {string} color - Hex color
     * @param {number} amount - Amount to adjust (-255 to 255)
     * @returns {string} Adjusted hex color
     */
    adjustColorBrightness(color, amount) {
        const hex = color.replace('#', '');
        const r = Math.max(0, Math.min(255, parseInt(hex.substr(0, 2), 16) + amount));
        const g = Math.max(0, Math.min(255, parseInt(hex.substr(2, 2), 16) + amount));
        const b = Math.max(0, Math.min(255, parseInt(hex.substr(4, 2), 16) + amount));
        return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
    }

    /**
     * Update the score display
     * @param {number} score - Current score
     */
    updateScoreDisplay(score) {
        this.scoreDisplay.textContent = score;
    }

    /**
     * Handle game over
     * @param {number} score - Final score
     */
    handleGameOver(score) {
        this.gameActive = false;
        this.currentScore = score;

        // Update final score display
        this.finalScoreDisplay.textContent = score;

        // Show game over overlay
        this.gameOverOverlay.classList.add('visible');

        // Focus the name input
        setTimeout(() => {
            this.playerNameInput.focus();
        }, 100);
    }

    /**
     * Submit score to leaderboard
     */
    async submitScore() {
        const name = this.playerNameInput.value.trim();

        if (!name) {
            alert('Please enter your name');
            this.playerNameInput.focus();
            return;
        }

        try {
            const response = await fetch('/leaderboard', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: name,
                    score: this.currentScore,
                    board_size: `${this.boardWidth}x${this.boardHeight}`
                })
            });

            if (!response.ok) {
                throw new Error('Failed to submit score');
            }

            // Refresh leaderboard
            await this.fetchLeaderboard();

            // Hide overlay and reset name input
            this.gameOverOverlay.classList.remove('visible');
            this.playerNameInput.value = '';

        } catch (error) {
            console.error('Error submitting score:', error);
            alert('Failed to submit score. Please try again.');
        }
    }

    /**
     * Start a new game (play again)
     */
    playAgain() {
        this.gameOverOverlay.classList.remove('visible');
        this.playerNameInput.value = '';
        
        // Restore previous mode and checkpoint if it was agent mode
        if (this.lastMode === 'agent') {
            this.setMode('agent');
            if (this.lastCheckpoint) {
                this.checkpointSelect.value = this.lastCheckpoint;
            }
        }
        
        this.startGame();
    }

    /**
     * Close the game over overlay without restarting
     */
    closeOverlay() {
        this.gameOverOverlay.classList.remove('visible');
        this.playerNameInput.value = '';
        this.gameActive = false;
    }

    /**
     * Fetch leaderboard from server
     */
    async fetchLeaderboard() {
        try {
            const response = await fetch('/leaderboard');
            if (!response.ok) {
                throw new Error('Failed to fetch leaderboard');
            }

            const data = await response.json();
            this.renderLeaderboard(data.leaderboard || []);
        } catch (error) {
            console.error('Error fetching leaderboard:', error);
        }
    }

    /**
     * Render the leaderboard
     * @param {Array} leaderboard - Array of {name, score} objects
     */
    renderLeaderboard(leaderboard) {
        if (!Array.isArray(leaderboard) || leaderboard.length === 0) {
            this.leaderboardList.innerHTML = `
                <li>
                    <span class="leaderboard-rank">-</span>
                    <span class="leaderboard-name">No scores yet</span>
                    <span class="leaderboard-score"></span>
                </li>
            `;
            return;
        }

        // Take top 10
        const top10 = leaderboard.slice(0, 10);

        this.leaderboardList.innerHTML = top10.map((entry, index) => `
            <li>
                <span class="leaderboard-rank">${index + 1}.</span>
                <span class="leaderboard-name">${this.escapeHtml(entry.name)}</span>
                <span class="leaderboard-score">${entry.score}</span>
            </li>
        `).join('');
    }

    /**
     * Escape HTML to prevent XSS
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize game when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.game = new GameController();
});
