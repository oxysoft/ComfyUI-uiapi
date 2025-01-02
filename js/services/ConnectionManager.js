import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";

/**
 * Class managing WebSocket connection and heartbeat
 */
export class ConnectionManager {
    constructor() {
        this.heartbeatInterval = 5000; // 5 seconds
        this.intervalId = null;
        this.isConnected = false;
    }

    /**
     * Initialize connection management
     */
    async initialize() {
        console.log("[ConnectionManager] Initializing...");
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Start heartbeat
        await this.startHeartbeat();
        
        // Perform initial connection
        await this.notifyServerConnection();
        
        console.log("[ConnectionManager] Initialization complete");
    }

    /**
     * Set up WebSocket event listeners
     * @private
     */
    setupEventListeners() {
        api.addEventListener("reconnected", async () => {
            console.log("[ConnectionManager] WebSocket reconnected - Notifying server...");
            await this.notifyServerConnection();
        });

        api.addEventListener("reconnecting", async () => {
            console.log("[ConnectionManager] WebSocket reconnecting - Setting disconnected state...");
            this.setDisconnected();
            await this.notifyServerDisconnect();
        });

        // Handle page unload
        window.addEventListener('beforeunload', async () => {
            console.log("[ConnectionManager] Page unloading - Cleaning up...");
            this.stopHeartbeat();
            await this.notifyServerDisconnect();
        });
    }

    /**
     * Start the heartbeat mechanism
     * @private
     */
    async startHeartbeat() {
        console.log("[ConnectionManager] Starting heartbeat mechanism...");
        
        // Clear any existing interval
        this.stopHeartbeat();

        // Start new heartbeat interval
        this.intervalId = setInterval(async () => {
            try {
                const response = await api.fetchApi("/uiapi/connection_status", {
                    method: "GET"
                });
                
                if (!response.ok) {
                    console.warn("[ConnectionManager] ⚠ Heartbeat check failed:", response.status);
                    return;
                }

                const data = await response.json();
                
                // If we weren't connected before but now we are, notify server
                if (!this.isConnected && data.webui_connected) {
                    console.log("[ConnectionManager] Connection restored - Notifying server...");
                    await this.notifyServerConnection();
                }
                
                // Update connection state
                this.setConnectionState(data.webui_connected);
                
                if (data.webui_connected) {
                    console.debug("[ConnectionManager] ♥ Heartbeat OK - Connection active");
                }

            } catch (err) {
                console.warn("[ConnectionManager] ⚠ Heartbeat check failed:", err);
                this.setDisconnected();
            }
        }, this.heartbeatInterval);

        console.log("[ConnectionManager] Heartbeat mechanism initialized");
    }

    /**
     * Stop the heartbeat mechanism
     * @private
     */
    stopHeartbeat() {
        if (this.intervalId) {
            console.log("[ConnectionManager] Clearing existing heartbeat interval");
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }

    /**
     * Notify server of client connection
     * @private
     */
    async notifyServerConnection() {
        try {
            await api.fetchApi("/uiapi/webui_ready", {
                method: "POST"
            });
            console.log("[ConnectionManager] ✓ Server successfully notified of connection");
            this.setConnected();
        } catch (err) {
            console.error("[ConnectionManager] ✗ Failed to notify server of connection:", err);
            this.setDisconnected();
        }
    }

    /**
     * Notify server of client disconnect
     * @private
     */
    async notifyServerDisconnect() {
        try {
            await api.fetchApi("/uiapi/client_disconnect", {
                method: "POST"
            });
            console.log("[ConnectionManager] ✓ Server notified of disconnect");
        } catch (err) {
            console.error("[ConnectionManager] ✗ Failed to notify server of disconnect:", err);
        }
    }

    /**
     * Set connection state
     * @private
     * @param {boolean} connected - Whether connected
     */
    setConnectionState(connected) {
        this.isConnected = connected;
        app.uiapi_connected = connected;
    }

    /**
     * Set to connected state
     * @private
     */
    setConnected() {
        this.setConnectionState(true);
    }

    /**
     * Set to disconnected state
     * @private
     */
    setDisconnected() {
        this.setConnectionState(false);
    }
}

// Create and export a singleton instance
export const connectionManager = new ConnectionManager(); 