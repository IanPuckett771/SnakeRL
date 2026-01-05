/**
 * WebSocket Manager for SnakeRL
 * Handles WebSocket connection, messaging, and reconnection logic
 */

class WebSocketManager {
    constructor() {
        this.socket = null;
        this.messageHandlers = [];
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.isConnecting = false;
        this.shouldReconnect = true;
    }

    /**
     * Get the WebSocket URL based on current host
     * @returns {string} WebSocket URL
     */
    getWebSocketUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws/game`;
    }

    /**
     * Connect to the WebSocket server
     * @returns {Promise} Resolves when connected, rejects on failure
     */
    connect() {
        return new Promise((resolve, reject) => {
            if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                resolve();
                return;
            }

            if (this.isConnecting) {
                reject(new Error('Connection already in progress'));
                return;
            }

            this.isConnecting = true;
            this.shouldReconnect = true;
            this.updateConnectionStatus('connecting');

            const url = this.getWebSocketUrl();
            console.log(`Connecting to WebSocket: ${url}`);

            try {
                this.socket = new WebSocket(url);
            } catch (error) {
                this.isConnecting = false;
                this.updateConnectionStatus('disconnected');
                reject(error);
                return;
            }

            this.socket.onopen = () => {
                console.log('WebSocket connected');
                this.isConnecting = false;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected');
                resolve();
            };

            this.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.messageHandlers.forEach(handler => handler(data));
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.isConnecting = false;
            };

            this.socket.onclose = (event) => {
                console.log(`WebSocket closed: ${event.code} - ${event.reason}`);
                this.isConnecting = false;
                this.updateConnectionStatus('disconnected');

                if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.attemptReconnect();
                }
            };

            // Timeout for initial connection
            setTimeout(() => {
                if (this.isConnecting) {
                    this.isConnecting = false;
                    if (this.socket) {
                        this.socket.close();
                    }
                    reject(new Error('Connection timeout'));
                }
            }, 5000);
        });
    }

    /**
     * Attempt to reconnect to the WebSocket server
     */
    attemptReconnect() {
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`);
        this.updateConnectionStatus('connecting');

        setTimeout(() => {
            if (this.shouldReconnect) {
                this.connect().catch(error => {
                    console.error('Reconnection failed:', error);
                });
            }
        }, delay);
    }

    /**
     * Send a message through the WebSocket
     * @param {Object} message - The message object to send
     * @returns {boolean} True if sent successfully, false otherwise
     */
    send(message) {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
            console.error('WebSocket is not connected');
            return false;
        }

        try {
            const jsonMessage = JSON.stringify(message);
            this.socket.send(jsonMessage);
            return true;
        } catch (error) {
            console.error('Error sending message:', error);
            return false;
        }
    }

    /**
     * Register a message handler callback
     * @param {Function} callback - Function to call when a message is received
     * @returns {Function} Function to unregister the handler
     */
    onMessage(callback) {
        if (typeof callback !== 'function') {
            throw new Error('Callback must be a function');
        }

        this.messageHandlers.push(callback);

        // Return unsubscribe function
        return () => {
            const index = this.messageHandlers.indexOf(callback);
            if (index > -1) {
                this.messageHandlers.splice(index, 1);
            }
        };
    }

    /**
     * Disconnect from the WebSocket server
     */
    disconnect() {
        this.shouldReconnect = false;

        if (this.socket) {
            this.socket.close(1000, 'Client disconnected');
            this.socket = null;
        }

        this.updateConnectionStatus('disconnected');
        console.log('WebSocket disconnected');
    }

    /**
     * Check if the WebSocket is currently connected
     * @returns {boolean} True if connected
     */
    isConnected() {
        return this.socket && this.socket.readyState === WebSocket.OPEN;
    }

    /**
     * Update the connection status indicator in the UI
     * @param {string} status - 'connected', 'disconnected', or 'connecting'
     */
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connectionStatus');
        if (!statusElement) return;

        statusElement.className = `connection-status ${status}`;

        switch (status) {
            case 'connected':
                statusElement.textContent = 'Connected';
                break;
            case 'disconnected':
                statusElement.textContent = 'Disconnected';
                break;
            case 'connecting':
                statusElement.textContent = 'Connecting...';
                break;
        }
    }
}

// Export singleton instance
const wsManager = new WebSocketManager();
