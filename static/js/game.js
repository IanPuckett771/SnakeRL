/**
 * Game Controller for SnakeRL Game Hub
 * Handles game rendering, input, and state management for multiple games
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
        this.currentGame = 'snake'; // 'snake', 'tron', 'tank'
        this.snakeColor = '#00ff00';
        this.boardWidth = 20;
        this.boardHeight = 20;

        // Tron-specific colors
        this.tronColors = {
            player1: '#00ffff', // Cyan
            player2: '#ff6600', // Orange
            background: '#0a0a1a'
        };

        // Tank-specific colors
        this.tankColors = {
            player: '#4a7c23', // Military green
            enemy: '#8b0000', // Dark red
            wall: '#4a4a4a', // Gray
            bullet: '#ffff00', // Yellow
            background: '#2d2d1f' // Camo-ish background
        };

        // UI Elements
        this.scoreDisplay = document.getElementById('currentScore');
        this.finalScoreDisplay = document.getElementById('finalScore');
        this.gameOverOverlay = document.getElementById('gameOverOverlay');
        this.leaderboardList = document.getElementById('leaderboardList');
        this.checkpointGroup = document.getElementById('checkpointGroup');
        this.checkpointSelect = document.getElementById('checkpoint');
        this.opponentGroup = document.getElementById('opponentGroup');
        this.opponentSelect = document.getElementById('opponentSelect');

        // Settings inputs
        this.boardWidthInput = document.getElementById('boardWidth');
        this.boardHeightInput = document.getElementById('boardHeight');
        this.snakeColorInput = document.getElementById('snakeColor');
        this.gameSpeedSelect = document.getElementById('gameSpeed');
        this.gameSelect = document.getElementById('gameSelect');

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
        // Keyboard controls - track held keys for tank game
        this.heldKeys = new Set();
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
        document.addEventListener('keyup', (e) => this.handleKeyUp(e));

        // Mode toggle buttons
        this.playModeBtn.addEventListener('click', () => this.setMode('play'));
        this.watchModeBtn.addEventListener('click', () => this.setMode('agent'));

        // Game selector
        if (this.gameSelect) {
            this.gameSelect.addEventListener('change', (e) => this.setGame(e.target.value));
        }

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
     * Set the current game
     * @param {string} game - 'snake', 'tron', or 'tank'
     */
    setGame(game) {
        this.currentGame = game;

        // Update canvas styling based on game
        this.updateCanvasStyle();

        // Update UI visibility based on game
        this.updateGameSettings();

        // Reconnect WebSocket with game parameter
        this.reconnectWithGame();

        // Re-fetch checkpoints for Watch Agent mode (filtered by game)
        if (this.currentMode === 'agent') {
            this.fetchCheckpoints();
        }

        // Redraw empty grid with new styling
        this.drawEmptyGrid();
    }

    /**
     * Update canvas styling based on current game
     */
    updateCanvasStyle() {
        // Remove all game-specific classes
        this.canvas.classList.remove('game-snake', 'game-tron', 'game-tank');

        // Add current game class
        this.canvas.classList.add(`game-${this.currentGame}`);
    }

    /**
     * Update settings visibility based on current game
     */
    updateGameSettings() {
        const snakeColorGroup = document.getElementById('snakeColorGroup');
        if (snakeColorGroup) {
            // Only show snake color for snake game
            snakeColorGroup.style.display = this.currentGame === 'snake' ? 'block' : 'none';
        }

        // Show opponent selector for Tron and Tank (2-player/vs-AI games)
        if (this.opponentGroup) {
            const showOpponent = this.currentGame === 'tron' || this.currentGame === 'tank';
            this.opponentGroup.classList.toggle('visible', showOpponent);

            // Fetch checkpoints for opponent selection
            if (showOpponent) {
                this.fetchOpponentCheckpoints();
            }
        }
    }

    /**
     * Fetch available checkpoints for opponent AI selection
     */
    async fetchOpponentCheckpoints() {
        try {
            const response = await fetch('/checkpoints');
            if (!response.ok) {
                throw new Error('Failed to fetch checkpoints');
            }
            const data = await response.json();
            this.populateOpponentDropdown(data.checkpoints || []);
        } catch (error) {
            console.error('Error fetching opponent checkpoints:', error);
            this.populateOpponentDropdown([]);
        }
    }

    /**
     * Populate the opponent dropdown with available checkpoints
     */
    populateOpponentDropdown(checkpoints) {
        if (!this.opponentSelect) return;

        // Clear existing options except the default
        this.opponentSelect.innerHTML = '<option value="" selected>Simple AI (Default)</option>';

        // Filter checkpoints relevant to current game
        const gameCheckpoints = checkpoints.filter(cp => {
            const cpLower = cp.toLowerCase();
            if (this.currentGame === 'tron') {
                return cpLower.includes('tron');
            } else if (this.currentGame === 'tank') {
                return cpLower.includes('tank');
            }
            return false;
        });

        // Add filtered checkpoints (only show checkpoints for current game)
        gameCheckpoints.forEach(checkpoint => {
            const option = document.createElement('option');
            option.value = checkpoint;
            option.textContent = checkpoint;
            this.opponentSelect.appendChild(option);
        });
    }

    /**
     * Reconnect WebSocket with game query parameter
     */
    reconnectWithGame() {
        // Disconnect current connection
        wsManager.disconnect();

        // Store original URL getter once, then override with game parameter
        if (!this._originalGetUrl) {
            this._originalGetUrl = wsManager.getWebSocketUrl.bind(wsManager);
        }

        const game = this.currentGame;
        wsManager.getWebSocketUrl = () => {
            const baseUrl = this._originalGetUrl();
            return `${baseUrl}?game=${game}`;
        };

        // Reconnect
        wsManager.connect().catch(error => {
            console.error('Failed to reconnect WebSocket:', error);
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

        // Normalize key
        const key = e.key.toLowerCase();

        if (this.currentGame === 'tank') {
            // Tank: track held keys for combined movement
            const tankKeys = ['w', 's', 'a', 'd', ' ', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'];
            if (tankKeys.includes(key)) {
                e.preventDefault();
                this.heldKeys.add(key);
                this.sendTankInput();
            }
        } else {
            // Snake/Tron controls: single direction per key press
            let action = null;
            switch (key) {
                case 'arrowup':
                case 'w':
                    action = 'up';
                    break;
                case 'arrowdown':
                case 's':
                    action = 'down';
                    break;
                case 'arrowleft':
                case 'a':
                    action = 'left';
                    break;
                case 'arrowright':
                case 'd':
                    action = 'right';
                    break;
            }

            if (action) {
                e.preventDefault();
                wsManager.send({
                    type: 'action',
                    action: action
                });
            }
        }
    }

    /**
     * Handle key release
     * @param {KeyboardEvent} e - Keyboard event
     */
    handleKeyUp(e) {
        const key = e.key.toLowerCase();
        this.heldKeys.delete(key);

        // Also remove arrow key aliases
        if (key === 'arrowup') this.heldKeys.delete('w');
        if (key === 'arrowdown') this.heldKeys.delete('s');
        if (key === 'arrowleft') this.heldKeys.delete('a');
        if (key === 'arrowright') this.heldKeys.delete('d');
        if (key === 'w') this.heldKeys.delete('arrowup');
        if (key === 's') this.heldKeys.delete('arrowdown');
        if (key === 'a') this.heldKeys.delete('arrowleft');
        if (key === 'd') this.heldKeys.delete('arrowright');

        // Send updated tank input
        if (this.currentGame === 'tank' && this.gameActive) {
            this.sendTankInput();
        }
    }

    /**
     * Send tank input based on currently held keys
     */
    sendTankInput() {
        // Build input state from held keys
        const forward = this.heldKeys.has('w') || this.heldKeys.has('arrowup');
        const backward = this.heldKeys.has('s') || this.heldKeys.has('arrowdown');
        const turnLeft = this.heldKeys.has('a') || this.heldKeys.has('arrowleft');
        const turnRight = this.heldKeys.has('d') || this.heldKeys.has('arrowright');
        const shoot = this.heldKeys.has(' ');

        wsManager.send({
            type: 'tank_input',
            forward: forward,
            backward: backward,
            turn_left: turnLeft,
            turn_right: turnRight,
            shoot: shoot
        });
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

            const data = await response.json();
            this.populateCheckpointDropdown(data.checkpoints || []);
        } catch (error) {
            console.error('Error fetching checkpoints:', error);
            this.checkpointSelect.innerHTML = '<option value="">No checkpoints available</option>';
        }
    }

    /**
     * Populate the checkpoint dropdown with options (filtered by current game)
     * @param {Array} checkpoints - Array of checkpoint objects
     */
    populateCheckpointDropdown(checkpoints) {
        this.checkpointSelect.innerHTML = '<option value="">Select a checkpoint...</option>';

        if (Array.isArray(checkpoints) && checkpoints.length > 0) {
            // Filter checkpoints by current game
            const gameCheckpoints = checkpoints.filter(cp => {
                const cpStr = (cp.path || cp.name || cp).toLowerCase();
                return cpStr.includes(this.currentGame);
            });

            if (gameCheckpoints.length > 0) {
                gameCheckpoints.forEach(checkpoint => {
                    const option = document.createElement('option');
                    option.value = checkpoint.path || checkpoint.name || checkpoint;
                    option.textContent = checkpoint.name || checkpoint.path || checkpoint;
                    this.checkpointSelect.appendChild(option);
                });
            } else {
                this.checkpointSelect.innerHTML = `<option value="">No ${this.currentGame} checkpoints available</option>`;
            }
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
            game: this.currentGame,
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

        // Add opponent checkpoint for Tron/Tank games
        if (this.currentGame === 'tron' || this.currentGame === 'tank') {
            const opponentCheckpoint = this.opponentSelect ? this.opponentSelect.value : '';
            if (opponentCheckpoint) {
                config.opponent_checkpoint = opponentCheckpoint;
            }
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

        // Dispatch to game-specific renderer
        switch (this.currentGame) {
            case 'tron':
                this.renderTron(state);
                break;
            case 'tank':
                this.renderTank(state);
                break;
            case 'snake':
            default:
                this.renderSnake(state);
                break;
        }
    }

    /**
     * Render Snake game state
     * @param {Object} state - Snake game state from server
     */
    renderSnake(state) {
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
     * Render Tron (light cycles) game state
     * @param {Object} state - Tron game state from server
     */
    renderTron(state) {
        const { width, height, players, score } = state;

        // Clear canvas with dark Tron background
        this.ctx.fillStyle = this.tronColors.background;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw grid with neon glow effect
        this.drawTronGrid(width, height);

        // Draw player trails and heads
        if (players) {
            players.forEach((player, index) => {
                const color = index === 0 ? this.tronColors.player1 : this.tronColors.player2;
                this.drawTronPlayer(player, color, index);
            });
        }

        // Update score
        if (score !== undefined) {
            this.currentScore = score;
            this.updateScoreDisplay(score);
        }
    }

    /**
     * Draw Tron-style grid with neon glow
     * @param {number} width - Grid width in cells
     * @param {number} height - Grid height in cells
     */
    drawTronGrid(width, height) {
        this.ctx.strokeStyle = 'rgba(0, 255, 255, 0.15)';
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
     * Draw a Tron player (light cycle) with trail
     * @param {Object} player - Player data with trail array
     * @param {string} color - Player color
     * @param {number} playerIndex - Player index (0 or 1)
     */
    drawTronPlayer(player, color, playerIndex) {
        if (!player || !player.trail || player.trail.length === 0) return;

        const trail = player.trail;

        // Draw trail with glow effect
        this.ctx.shadowColor = color;
        this.ctx.shadowBlur = 10;

        // Draw trail segments
        trail.forEach((segment, index) => {
            const x = segment.x * this.cellSize;
            const y = segment.y * this.cellSize;
            const isHead = index === trail.length - 1;

            if (isHead) {
                // Draw head as a brighter, larger shape
                this.ctx.fillStyle = color;
                this.ctx.shadowBlur = 20;

                // Draw diamond shape for head
                const centerX = x + this.cellSize / 2;
                const centerY = y + this.cellSize / 2;
                const size = this.cellSize / 2 - 2;

                this.ctx.beginPath();
                this.ctx.moveTo(centerX, centerY - size);
                this.ctx.lineTo(centerX + size, centerY);
                this.ctx.lineTo(centerX, centerY + size);
                this.ctx.lineTo(centerX - size, centerY);
                this.ctx.closePath();
                this.ctx.fill();

                // Inner glow for head
                this.ctx.fillStyle = '#ffffff';
                this.ctx.shadowBlur = 5;
                const innerSize = size * 0.4;
                this.ctx.beginPath();
                this.ctx.moveTo(centerX, centerY - innerSize);
                this.ctx.lineTo(centerX + innerSize, centerY);
                this.ctx.lineTo(centerX, centerY + innerSize);
                this.ctx.lineTo(centerX - innerSize, centerY);
                this.ctx.closePath();
                this.ctx.fill();
            } else {
                // Draw trail segment
                this.ctx.fillStyle = this.adjustColorBrightness(color, -40);
                this.ctx.shadowBlur = 8;
                this.ctx.fillRect(x + 2, y + 2, this.cellSize - 4, this.cellSize - 4);
            }
        });

        // Reset shadow
        this.ctx.shadowBlur = 0;
    }

    /**
     * Render Tank battle game state (smooth physics version)
     * @param {Object} state - Tank game state from server
     */
    renderTank(state) {
        const { width, height, player, enemies, walls, bullets, collectibles, health, score } = state;

        // Resize canvas to match arena (positions are in pixels now)
        if (this.canvas.width !== width || this.canvas.height !== height) {
            this.canvas.width = width;
            this.canvas.height = height;
        }

        // Clear canvas with dark background
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, width, height);

        // Draw ground texture/pattern
        this.drawTankGround(width, height);

        // Draw walls/obstacles
        if (walls && walls.length > 0) {
            this.drawTankWalls(walls);
        }

        // Draw collectibles
        if (collectibles && collectibles.length > 0) {
            collectibles.forEach(c => this.drawCollectible(c));
        }

        // Draw bullets with glow
        if (bullets && bullets.length > 0) {
            this.drawTankBullets(bullets);
        }

        // Draw enemy tanks
        if (enemies && enemies.length > 0) {
            enemies.forEach(enemy => {
                this.drawSmoothTank(enemy.x, enemy.y, enemy.angle, '#aa2222', false, enemy.health);
            });
        }

        // Draw player tank
        if (player) {
            this.drawSmoothTank(player.x, player.y, player.angle, '#22aa22', true, player.health);
        }

        // Draw HUD
        this.drawTankHUD(health, score, width, height);
    }

    /**
     * Draw tank terrain for empty grid (before game starts)
     */
    drawTankTerrain(width, height) {
        // Simple checkered pattern
        this.ctx.fillStyle = '#353525';
        for (let x = 0; x < width; x++) {
            for (let y = 0; y < height; y++) {
                if ((x + y) % 2 === 0) {
                    this.ctx.fillRect(x * this.cellSize, y * this.cellSize, this.cellSize, this.cellSize);
                }
            }
        }
    }

    /**
     * Draw ground texture for tank arena
     */
    drawTankGround(width, height) {
        // Subtle noise pattern
        this.ctx.fillStyle = '#252520';
        for (let x = 0; x < width; x += 40) {
            for (let y = 0; y < height; y += 40) {
                if ((x + y) % 80 === 0) {
                    this.ctx.fillRect(x, y, 40, 40);
                }
            }
        }

        // Border
        this.ctx.strokeStyle = '#444';
        this.ctx.lineWidth = 4;
        this.ctx.strokeRect(2, 2, width - 4, height - 4);
    }

    /**
     * Draw tank walls/obstacles (smooth version with actual dimensions)
     */
    drawTankWalls(walls) {
        walls.forEach(wall => {
            // Wall shadow
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            this.ctx.fillRect(wall.x + 4, wall.y + 4, wall.width, wall.height);

            // Wall body with gradient
            const gradient = this.ctx.createLinearGradient(wall.x, wall.y, wall.x, wall.y + wall.height);
            gradient.addColorStop(0, '#666');
            gradient.addColorStop(0.5, '#555');
            gradient.addColorStop(1, '#444');
            this.ctx.fillStyle = gradient;
            this.ctx.fillRect(wall.x, wall.y, wall.width, wall.height);

            // Top highlight
            this.ctx.fillStyle = '#888';
            this.ctx.fillRect(wall.x, wall.y, wall.width, 3);

            // Left highlight
            this.ctx.fillRect(wall.x, wall.y, 3, wall.height);

            // Border
            this.ctx.strokeStyle = '#333';
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(wall.x, wall.y, wall.width, wall.height);
        });
    }

    /**
     * Draw bullets with glow effect (smooth positions)
     */
    drawTankBullets(bullets) {
        bullets.forEach(bullet => {
            // Glow
            this.ctx.shadowColor = bullet.owner === 0 ? '#00ff00' : '#ff6600';
            this.ctx.shadowBlur = 10;

            // Bullet trail
            const trailLength = 8;
            const dx = Math.cos(bullet.angle) * trailLength;
            const dy = Math.sin(bullet.angle) * trailLength;

            this.ctx.strokeStyle = bullet.owner === 0 ? '#88ff88' : '#ff8844';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(bullet.x - dx, bullet.y - dy);
            this.ctx.lineTo(bullet.x, bullet.y);
            this.ctx.stroke();

            // Bullet core
            this.ctx.fillStyle = bullet.owner === 0 ? '#ffff00' : '#ff4400';
            this.ctx.beginPath();
            this.ctx.arc(bullet.x, bullet.y, 4, 0, Math.PI * 2);
            this.ctx.fill();

            this.ctx.shadowBlur = 0;
        });
    }

    /**
     * Draw a tank with smooth rotation (angle in radians)
     */
    drawSmoothTank(x, y, angle, color, isPlayer, health) {
        this.ctx.save();
        this.ctx.translate(x, y);
        this.ctx.rotate(angle + Math.PI / 2); // Adjust so 0 = pointing up

        const size = 24; // Tank size in pixels

        // Tank shadow
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        this.ctx.beginPath();
        this.ctx.ellipse(3, 3, size * 0.6, size * 0.5, 0, 0, Math.PI * 2);
        this.ctx.fill();

        // Tank body glow
        if (isPlayer) {
            this.ctx.shadowColor = '#00ff00';
            this.ctx.shadowBlur = 15;
        }

        // Tank tracks
        this.ctx.fillStyle = '#222';
        this.ctx.fillRect(-size * 0.7, -size * 0.5, size * 0.25, size);
        this.ctx.fillRect(size * 0.45, -size * 0.5, size * 0.25, size);

        // Track details
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 1;
        for (let i = -4; i <= 4; i++) {
            const ty = i * (size / 5);
            this.ctx.beginPath();
            this.ctx.moveTo(-size * 0.7, ty);
            this.ctx.lineTo(-size * 0.45, ty);
            this.ctx.stroke();
            this.ctx.beginPath();
            this.ctx.moveTo(size * 0.45, ty);
            this.ctx.lineTo(size * 0.7, ty);
            this.ctx.stroke();
        }

        // Tank body
        const gradient = this.ctx.createRadialGradient(0, 0, 0, 0, 0, size);
        gradient.addColorStop(0, this.adjustColorBrightness(color, 40));
        gradient.addColorStop(1, color);
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.roundRect(-size * 0.4, -size * 0.45, size * 0.8, size * 0.9, 4);
        this.ctx.fill();

        // Turret base
        this.ctx.fillStyle = this.adjustColorBrightness(color, -20);
        this.ctx.beginPath();
        this.ctx.arc(0, 0, size * 0.3, 0, Math.PI * 2);
        this.ctx.fill();

        // Cannon
        this.ctx.fillStyle = '#333';
        this.ctx.fillRect(-3, -size * 0.8, 6, size * 0.5);

        // Cannon tip
        this.ctx.fillStyle = '#444';
        this.ctx.fillRect(-4, -size * 0.85, 8, 6);

        this.ctx.shadowBlur = 0;

        // Health indicator (small bar above tank)
        if (health !== undefined && health < 3) {
            this.ctx.fillStyle = '#333';
            this.ctx.fillRect(-15, -size - 8, 30, 4);
            const healthPct = health / 3;
            this.ctx.fillStyle = healthPct > 0.5 ? '#00ff00' : healthPct > 0.25 ? '#ffff00' : '#ff0000';
            this.ctx.fillRect(-15, -size - 8, 30 * healthPct, 4);
        }

        this.ctx.restore();
    }

    /**
     * Draw collectible power-up
     */
    drawCollectible(collectible) {
        const { x, y, type } = collectible;

        // Glow
        this.ctx.shadowColor = type === 'health' ? '#00ff00' : '#ffff00';
        this.ctx.shadowBlur = 15;

        // Pulsing effect
        const pulse = 1 + Math.sin(Date.now() / 200) * 0.1;
        const radius = 10 * pulse;

        // Outer ring
        this.ctx.strokeStyle = type === 'health' ? '#00ff00' : '#ffff00';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
        this.ctx.stroke();

        // Inner circle
        this.ctx.fillStyle = type === 'health' ? '#44ff44' : '#ffff44';
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fill();

        // Icon
        this.ctx.fillStyle = '#fff';
        this.ctx.font = 'bold 12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(type === 'health' ? '+' : '★', x, y);

        this.ctx.shadowBlur = 0;
    }

    /**
     * Draw tank game HUD
     */
    drawTankHUD(health, score, width, height) {
        // Health bar at top-left
        const barWidth = 100;
        const barHeight = 12;
        const barX = 15;
        const barY = 15;

        // Background
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(barX - 2, barY - 2, barWidth + 4, barHeight + 4);

        // Health bar background
        this.ctx.fillStyle = '#333';
        this.ctx.fillRect(barX, barY, barWidth, barHeight);

        // Health bar fill
        const healthPct = (health || 0) / 3;
        const healthColor = healthPct > 0.5 ? '#00cc00' : healthPct > 0.25 ? '#cccc00' : '#cc0000';
        this.ctx.fillStyle = healthColor;
        this.ctx.fillRect(barX, barY, barWidth * healthPct, barHeight);

        // Health text
        this.ctx.fillStyle = '#fff';
        this.ctx.font = 'bold 10px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(`${health || 0}/3`, barX + barWidth / 2, barY + barHeight / 2);

        // Score at top-right
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(width - 85, barY - 2, 70, barHeight + 4);
        this.ctx.fillStyle = '#ffcc00';
        this.ctx.font = 'bold 12px Arial';
        this.ctx.textAlign = 'right';
        this.ctx.fillText(`Score: ${score || 0}`, width - 20, barY + barHeight / 2);

        // Update main score display
        if (score !== undefined) {
            this.currentScore = score;
            this.updateScoreDisplay(score);
        }
    }

    /**
     * Draw health bar for player tank
     * @param {number} health - Current health (0-100)
     */
    drawHealthBar(health) {
        const barWidth = 150;
        const barHeight = 15;
        const x = 10;
        const y = 10;
        const healthPercent = Math.max(0, Math.min(100, health)) / 100;

        // Background
        this.ctx.fillStyle = '#333';
        this.ctx.fillRect(x, y, barWidth, barHeight);

        // Health fill (green to red gradient based on health)
        const healthColor = healthPercent > 0.5
            ? `rgb(${Math.round(255 * (1 - healthPercent) * 2)}, 255, 0)`
            : `rgb(255, ${Math.round(255 * healthPercent * 2)}, 0)`;
        this.ctx.fillStyle = healthColor;
        this.ctx.fillRect(x + 2, y + 2, (barWidth - 4) * healthPercent, barHeight - 4);

        // Border
        this.ctx.strokeStyle = '#fff';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(x, y, barWidth, barHeight);

        // Health text
        this.ctx.fillStyle = '#fff';
        this.ctx.font = '10px "Press Start 2P", monospace';
        this.ctx.fillText(`HP: ${Math.round(health)}`, x + barWidth + 10, y + 12);
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

        // Use game-specific background
        switch (this.currentGame) {
            case 'tron':
                this.ctx.fillStyle = this.tronColors.background;
                this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
                this.drawTronGrid(width, height);
                break;
            case 'tank':
                this.ctx.fillStyle = this.tankColors.background;
                this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
                this.drawTankTerrain(width, height);
                break;
            case 'snake':
            default:
                this.ctx.fillStyle = '#000';
                this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
                this.drawGrid(width, height);
                break;
        }
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

        // Only show leaderboard overlay for human players, not agents
        if (this.currentMode === 'play') {
            // Show game over overlay
            this.gameOverOverlay.classList.add('visible');

            // Focus the name input
            setTimeout(() => {
                this.playerNameInput.focus();
            }, 100);
        }
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
