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
        this.checkpointList = document.getElementById('checkpointList');
        this.checkpointInfo = document.getElementById('checkpointInfo');
        this.checkpointName = document.getElementById('checkpointName');
        this.checkpointMeta = document.getElementById('checkpointMeta');
        this.trainingProgress = document.getElementById('trainingProgress');
        this.progressBar = document.getElementById('progressBar');
        this.trainingStats = document.getElementById('trainingStats');
        
        // Track last checkpoint used for "Play Again"
        this.lastCheckpoint = null;
        this.lastMode = 'play';
        this._fetchingCheckpoints = false;

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

        // Click outside the game-over content to dismiss
        this.gameOverOverlay.addEventListener('click', (e) => {
            if (e.target === this.gameOverOverlay) {
                this.closeOverlay();
            }
        });

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

        // Live speed change — send to server without resetting
        this.gameSpeedSelect.addEventListener('change', () => {
            const speed = parseFloat(this.gameSpeedSelect.value) || 0.15;
            if (wsManager.isConnected()) {
                wsManager.send({ type: 'set_speed', speed });
            }
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
     * Get the currently selected checkpoint value
     * @returns {string} Checkpoint filename or empty string
     */
    get selectedCheckpoint() {
        const card = this.checkpointList.querySelector('.checkpoint-card.selected');
        return card ? card.dataset.value : '';
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

        // Show/hide checkpoint list
        this.checkpointGroup.classList.toggle('visible', mode === 'agent');

        // Fetch checkpoints if switching to agent mode
        if (mode === 'agent') {
            this.fetchCheckpoints();
        }

        // Update start button text
        this.startBtn.textContent = mode === 'play' ? 'Start Game' : 'Watch Agent';
    }

    /**
     * Select a checkpoint card by data-value
     * @param {string} value - The checkpoint value to select
     */
    selectCheckpointCard(value) {
        const cards = this.checkpointList.querySelectorAll('.checkpoint-card');
        let found = false;
        cards.forEach(card => {
            if (card.dataset.value === value) {
                card.classList.add('selected');
                found = true;
            } else {
                card.classList.remove('selected');
            }
        });
        return found;
    }

    /**
     * Fetch available checkpoints from the server
     */
    async fetchCheckpoints() {
        if (this._fetchingCheckpoints) return;
        this._fetchingCheckpoints = true;

        try {
            // Preserve current selection before repopulating
            const currentSelection = this.selectedCheckpoint;

            const response = await fetch('/checkpoints');
            if (!response.ok) {
                throw new Error('Failed to fetch checkpoints');
            }

            const data = await response.json();
            const checkpoints = data.checkpoints || data;
            this.populateCheckpointList(checkpoints, data.best_replay || null);

            // Restore selection if it still exists
            if (currentSelection) {
                this.selectCheckpointCard(currentSelection);
            }
        } catch (error) {
            console.error('Error fetching checkpoints:', error);
            this.checkpointList.innerHTML = `
                <div class="checkpoint-card selected" data-value="">
                    <div class="radio"></div>
                    <div class="checkpoint-card-body">
                        <div class="checkpoint-card-title">No checkpoints available</div>
                    </div>
                </div>`;
        } finally {
            this._fetchingCheckpoints = false;
        }
    }

    /**
     * Create a checkpoint card element
     * @param {string} value - data-value for the card
     * @param {string} title - Card title text
     * @param {string} subtitle - Card subtitle text
     * @param {string} badgeClass - Optional badge CSS class
     * @param {string} badgeText - Optional badge text
     * @returns {HTMLElement} The card element
     */
    createCheckpointCard(value, title, subtitle, badgeClass, badgeText) {
        const card = document.createElement('div');
        card.className = 'checkpoint-card';
        card.dataset.value = value;

        const radio = document.createElement('div');
        radio.className = 'radio';

        const body = document.createElement('div');
        body.className = 'checkpoint-card-body';

        const titleEl = document.createElement('div');
        titleEl.className = 'checkpoint-card-title';
        titleEl.textContent = title;
        body.appendChild(titleEl);

        if (subtitle) {
            const subEl = document.createElement('div');
            subEl.className = 'checkpoint-card-subtitle';
            subEl.textContent = subtitle;
            body.appendChild(subEl);
        }

        card.appendChild(radio);
        card.appendChild(body);

        if (badgeClass && badgeText) {
            const badge = document.createElement('div');
            badge.className = `checkpoint-card-badge ${badgeClass}`;
            badge.textContent = badgeText;
            card.appendChild(badge);
        }

        card.addEventListener('click', () => {
            this.checkpointList.querySelectorAll('.checkpoint-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
        });

        return card;
    }

    /**
     * Populate the checkpoint list with card-based UI.
     * Groups by training run, shows metadata, auto-selects best.
     * @param {Array} checkpoints - Array of checkpoint objects
     * @param {Object} bestReplay - Best replay metadata or null
     */
    populateCheckpointList(checkpoints, bestReplay) {
        this.checkpointList.innerHTML = '';

        // Determine what will be auto-selected (set later)
        let autoSelectValue = '';

        // --- Parse checkpoints into structured data ---
        const runs = {};
        const mainCheckpoints = [];
        const peakCheckpoints = [];

        if (Array.isArray(checkpoints)) {
            checkpoints.forEach(cp => {
                const name = typeof cp === 'string' ? cp : (cp.path || cp.name || cp);
                const meta = (typeof cp === 'object' && cp.meta) ? cp.meta : null;

                if (!name.endsWith('.pt') || name === 'default_agent.pt') return;

                const peakMatch = name.match(/^(.+?)_peak_best\.pt$/);
                if (peakMatch) {
                    peakCheckpoints.push({ value: name, algo: peakMatch[1].toUpperCase(), meta });
                    return;
                }

                const tsMatch = name.match(/^(.+?)_agent_(\d{8})_(\d{6})_stage(\d+)\.pt$/);
                if (tsMatch) {
                    const algo = tsMatch[1].toUpperCase();
                    const dateStr = tsMatch[2];
                    const timeStr = tsMatch[3];
                    const ts = `${dateStr}_${timeStr}`;
                    const stageNum = parseInt(tsMatch[4]);

                    if (!runs[ts]) {
                        runs[ts] = { algo, dateStr, timeStr, stages: [] };
                    }
                    runs[ts].stages.push({ value: name, stage: stageNum, meta });
                    return;
                }

                const mainMatch = name.match(/^(.+?)_agent\.pt$/);
                if (mainMatch) {
                    mainCheckpoints.push({ value: name, algo: mainMatch[1].toUpperCase(), meta });
                    return;
                }
            });
        }

        const fmtMeta = (meta) => {
            if (!meta) return '';
            const parts = [];
            if (meta.avg_snake_length != null) parts.push(`Avg: ${meta.avg_snake_length}`);
            if (meta.max_snake_length != null) parts.push(`Max: ${meta.max_snake_length}`);
            return parts.length > 0 ? parts.join(', ') : '';
        };

        // Short label for avg snake length in card titles
        const avgLabel = (meta) => {
            if (!meta || meta.avg_snake_length == null) return '';
            return ` (Avg ${meta.avg_snake_length})`;
        };

        const sortedTimestamps = Object.keys(runs).sort().reverse();

        // --- Determine auto-select: latest run's highest stage > peak > main > default ---
        if (sortedTimestamps.length > 0) {
            const latestRun = runs[sortedTimestamps[0]];
            latestRun.stages.sort((a, b) => a.stage - b.stage);
            autoSelectValue = latestRun.stages[latestRun.stages.length - 1].value;
        } else if (peakCheckpoints.length > 0) {
            autoSelectValue = peakCheckpoints[0].value;
        } else if (mainCheckpoints.length > 0) {
            autoSelectValue = mainCheckpoints[0].value;
        }

        // --- Latest Run (top section) ---
        if (sortedTimestamps.length > 0) {
            const ts = sortedTimestamps[0];
            const run = runs[ts];
            run.stages.sort((a, b) => a.stage - b.stage);
            const bestStage = run.stages[run.stages.length - 1];

            const header = document.createElement('div');
            header.className = 'checkpoint-section-header';
            header.textContent = 'Latest Run';
            this.checkpointList.appendChild(header);

            // Show the best stage from latest run as the recommended card
            const metaStr = fmtMeta(bestStage.meta);
            const subtitle = `Stage ${bestStage.stage}/10` + (metaStr ? ` · ${metaStr}` : '');
            const card = this.createCheckpointCard(
                bestStage.value,
                `${run.algo} - Latest${avgLabel(bestStage.meta)}`,
                subtitle,
                'badge-recommended',
                'Best'
            );
            this.checkpointList.appendChild(card);

            // If there are other stages in the latest run, show them
            if (run.stages.length > 1) {
                const otherStages = run.stages.slice(0, -1).reverse();
                otherStages.forEach(item => {
                    const ms = fmtMeta(item.meta);
                    const sub = `Stage ${item.stage}/10` + (ms ? ` · ${ms}` : '');
                    const c = this.createCheckpointCard(item.value, `${run.algo} - Stage ${item.stage}${avgLabel(item.meta)}`, sub);
                    this.checkpointList.appendChild(c);
                });
            }
        }

        // --- Peak Performance ---
        if (peakCheckpoints.length > 0) {
            const header = document.createElement('div');
            header.className = 'checkpoint-section-header';
            header.textContent = 'Peak Performance';
            this.checkpointList.appendChild(header);

            peakCheckpoints.forEach(cp => {
                const metaStr = fmtMeta(cp.meta);
                const card = this.createCheckpointCard(
                    cp.value,
                    `${cp.algo} Peak Best${avgLabel(cp.meta)}`,
                    metaStr || 'All-time best performance',
                    'badge-peak',
                    'Peak'
                );
                this.checkpointList.appendChild(card);
            });
        }

        // --- Best Replay ---
        if (bestReplay) {
            const header = document.createElement('div');
            header.className = 'checkpoint-section-header';
            header.textContent = 'Replay';
            this.checkpointList.appendChild(header);

            const parts = [];
            if (bestReplay.length != null) parts.push(`Length: ${bestReplay.length}`);
            if (bestReplay.score != null) parts.push(`Score: ${bestReplay.score}`);
            const card = this.createCheckpointCard(
                '__best_replay__',
                'Watch Best Game Ever',
                parts.join(' · ') || 'Recorded best game',
                'badge-replay',
                'Replay'
            );
            this.checkpointList.appendChild(card);
        }

        // --- Current Model ---
        if (mainCheckpoints.length > 0) {
            const header = document.createElement('div');
            header.className = 'checkpoint-section-header';
            header.textContent = 'Current Model';
            this.checkpointList.appendChild(header);

            mainCheckpoints.forEach(cp => {
                const metaStr = fmtMeta(cp.meta);
                const card = this.createCheckpointCard(
                    cp.value,
                    `${cp.algo} - Current Best${avgLabel(cp.meta)}`,
                    metaStr || cp.value
                );
                this.checkpointList.appendChild(card);
            });
        }

        // --- Default Agent (always at bottom) ---
        const defaultCard = this.createCheckpointCard(
            '',
            'Default Agent (Heuristic)',
            'Rule-based fallback'
        );
        this.checkpointList.appendChild(defaultCard);

        // --- Earlier Runs (collapsed) ---
        if (sortedTimestamps.length > 1) {
            const earlierRuns = sortedTimestamps.slice(1);
            const totalEarlier = earlierRuns.reduce((sum, ts) => sum + runs[ts].stages.length, 0);

            const toggle = document.createElement('div');
            toggle.className = 'checkpoint-section-toggle';
            toggle.innerHTML = `<span class="arrow">&#9654;</span> Earlier Runs (${totalEarlier})`;

            const itemsContainer = document.createElement('div');
            itemsContainer.className = 'checkpoint-section-items';

            toggle.addEventListener('click', () => {
                toggle.classList.toggle('expanded');
                itemsContainer.classList.toggle('expanded');
            });

            this.checkpointList.appendChild(toggle);
            this.checkpointList.appendChild(itemsContainer);

            const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

            earlierRuns.forEach(ts => {
                const run = runs[ts];
                run.stages.sort((a, b) => b.stage - a.stage);

                const month = parseInt(run.dateStr.substring(4, 6));
                const day = parseInt(run.dateStr.substring(6, 8));
                const hour = parseInt(run.timeStr.substring(0, 2));
                const min = run.timeStr.substring(2, 4);
                const ampm = hour >= 12 ? 'PM' : 'AM';
                const hour12 = hour % 12 || 12;
                const dateLabel = `${monthNames[month-1]} ${day} ${hour12}:${min}${ampm}`;

                run.stages.forEach(item => {
                    const ms = fmtMeta(item.meta);
                    const sub = `${dateLabel} · Stage ${item.stage}/10` + (ms ? ` · ${ms}` : '');
                    const card = this.createCheckpointCard(item.value, `${run.algo} Stage ${item.stage}${avgLabel(item.meta)}`, sub);
                    itemsContainer.appendChild(card);
                });
            });
        }

        // --- Auto-select ---
        this.selectCheckpointCard(autoSelectValue);
    }

    /**
     * Start a new game
     */
    startGame() {
        // Stop any active replay
        this.stopReplay();
        
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
            const checkpoint = this.selectedCheckpoint;
            
            // Special case: Best Replay mode
            if (checkpoint === '__best_replay__') {
                this.lastCheckpoint = '__best_replay__';
                this.lastMode = 'agent';
                this.playBestReplay();
                return;
            }
            
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
     * Play back the best recorded game replay
     */
    async playBestReplay() {
        try {
            // Show loading state
            this.checkpointInfo.style.display = 'block';
            this.checkpointName.textContent = '★ Best Game Ever';
            this.checkpointMeta.textContent = 'Loading replay...';
            
            const response = await fetch('/best-replay');
            if (!response.ok) {
                this.checkpointMeta.textContent = 'No replay saved yet';
                return;
            }
            const data = await response.json();
            if (data.error) {
                this.checkpointMeta.textContent = data.error;
                return;
            }
            
            const frames = data.frames;
            if (!frames || frames.length === 0) {
                this.checkpointMeta.textContent = 'Empty replay';
                return;
            }
            
            this.checkpointMeta.textContent = `Snake Length: ${data.snake_length} • Score: ${data.score} • ${frames.length} moves (best of ${data.total_games_tested} games)`;
            
            // Set canvas size from first frame
            this.resizeCanvas(frames[0].width || 20, frames[0].height || 20);
            this.gameOverOverlay.classList.remove('visible');
            this.gameActive = true;
            this.currentScore = 0;
            this.updateScoreDisplay(0);
            
            // Play through frames
            this._replayActive = true;
            for (let i = 0; i < frames.length; i++) {
                if (!this._replayActive) break;

                const frame = frames[i];
                this.renderState(frame);
                this.currentScore = frame.score || 0;
                this.updateScoreDisplay(this.currentScore);

                // Update progress in metadata
                const snakeLen = frame.snake ? frame.snake.length : 0;
                const pct = Math.round((i / frames.length) * 100);
                this.checkpointMeta.textContent = `Snake: ${snakeLen} • Score: ${frame.score || 0} • Frame ${i+1}/${frames.length} (${pct}%)`;

                // Read speed live so user can change it mid-replay
                const speed = parseFloat(this.gameSpeedSelect.value) || 0.15;
                await new Promise(resolve => setTimeout(resolve, speed * 1000));
            }
            
            // Show game over
            const lastFrame = frames[frames.length - 1];
            this.handleGameOver(lastFrame.score || data.score);
            
        } catch (error) {
            console.error('Error playing best replay:', error);
            this.checkpointMeta.textContent = 'Error loading replay';
        }
    }
    
    /**
     * Stop any active replay
     */
    stopReplay() {
        this._replayActive = false;
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
    async playAgain() {
        this.gameOverOverlay.classList.remove('visible');
        this.playerNameInput.value = '';

        // Restore previous mode and checkpoint if it was agent mode
        if (this.lastMode === 'agent') {
            this.currentMode = 'agent';
            this.playModeBtn.classList.remove('active');
            this.watchModeBtn.classList.add('active');
            this.checkpointGroup.classList.add('visible');
            this.startBtn.textContent = 'Watch Agent';

            // Fetch and wait for it to complete before restoring selection
            await this.fetchCheckpoints();
            if (this.lastCheckpoint) {
                this.selectCheckpointCard(this.lastCheckpoint);
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
