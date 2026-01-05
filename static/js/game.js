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
        this.currentMode = mode;

        // Update button states
        this.playModeBtn.classList.toggle('active', mode === 'play');
        this.watchModeBtn.classList.toggle('active', mode === 'agent');

        // Show/hide checkpoint dropdown
        this.checkpointGroup.classList.toggle('visible', mode === 'agent');

        // Fetch checkpoints if switching to agent mode
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
            const response = await fetch('/checkpoints');
            if (!response.ok) {
                throw new Error('Failed to fetch checkpoints');
            }

            const checkpoints = await response.json();
            this.populateCheckpointDropdown(checkpoints);
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
        this.checkpointSelect.innerHTML = '<option value="">Select a checkpoint...</option>';

        if (Array.isArray(checkpoints) && checkpoints.length > 0) {
            checkpoints.forEach(checkpoint => {
                const option = document.createElement('option');
                option.value = checkpoint.path || checkpoint.name || checkpoint;
                option.textContent = checkpoint.name || checkpoint.path || checkpoint;
                this.checkpointSelect.appendChild(option);
            });
        } else {
            this.checkpointSelect.innerHTML = '<option value="">No checkpoints available</option>';
        }
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

        // Add checkpoint if in agent mode
        if (this.currentMode === 'agent') {
            const checkpoint = this.checkpointSelect.value;
            if (!checkpoint) {
                alert('Please select a checkpoint for the agent');
                return;
            }
            config.checkpoint = checkpoint;
        }

        // Hide game over overlay if visible
        this.gameOverOverlay.classList.remove('visible');

        // Send start game message
        wsManager.send(config);

        // Reset score display
        this.currentScore = 0;
        this.updateScoreDisplay(0);
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

        const { width, height, snake, food, score } = state;

        // Clear canvas
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw grid lines
        this.drawGrid(width, height);

        // Draw food
        if (food) {
            this.drawFood(food);
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
     * Draw the food
     * @param {Object} food - Food position {x, y}
     */
    drawFood(food) {
        const x = food.x * this.cellSize;
        const y = food.y * this.cellSize;
        const centerX = x + this.cellSize / 2;
        const centerY = y + this.cellSize / 2;
        const radius = (this.cellSize - 4) / 2;

        // Draw glowing food
        this.ctx.shadowColor = '#ff0000';
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
        gradient.addColorStop(0, '#ff6666');
        gradient.addColorStop(0.5, '#ff0000');
        gradient.addColorStop(1, '#cc0000');

        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        this.ctx.fill();

        // Reset shadow
        this.ctx.shadowBlur = 0;
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
