import { api } from "../../../scripts/api.js";
import { showDownloadUrlDialog } from "../components/DownloadDialog.js";

/**
 * Class managing model downloads and status checks
 */
export class DownloadManager {
    constructor() {
        this.checkInterval = 2000; // 2 seconds
        this.activeChecks = new Set();
    }

    /**
     * Start downloading models from a download table
     * @param {Object} downloadTable - Table of models to download
     * @returns {Promise<Response>} API response
     */
    async downloadModels(downloadTable) {
        try {
            return await api.fetchApi('/uiapi/download_models', {
                method: 'POST',
                body: JSON.stringify({ download_table: downloadTable })
            });
        } catch (err) {
            console.error("[DownloadManager] Error starting download:", err);
            throw err;
        }
    }

    /**
     * Check the status of a download task
     * @param {string} taskId - ID of the download task
     */
    async checkDownloadStatus(taskId) {
        if (this.activeChecks.has(taskId)) {
            return; // Already checking this task
        }

        this.activeChecks.add(taskId);

        try {
            while (this.activeChecks.has(taskId)) {
                const response = await api.fetchApi(`/uiapi/download_status/${taskId}`, {
                    method: "GET"
                });

                if (!response.ok) {
                    console.warn("[DownloadManager] Failed to check download status:", response.status);
                    break;
                }

                const data = await response.json();
                await this.handleStatusData(data);

                // Break the loop if download is complete
                if (data.status !== 'downloading' && data.status !== 'pending') {
                    break;
                }

                // Wait before next check
                await new Promise(resolve => setTimeout(resolve, this.checkInterval));
            }
        } catch (err) {
            console.error("[DownloadManager] Error checking download status:", err);
        } finally {
            this.activeChecks.delete(taskId);
        }
    }

    /**
     * Handle download status data
     * @private
     * @param {Object} data - Status data from API
     */
    async handleStatusData(data) {
        if (!data.progress) return;

        for (const [ckpt, info] of Object.entries(data.progress)) {
            if (info.status === 'waiting_for_url') {
                const modelName = ckpt.split('/').pop();
                const downloadUrl = await showDownloadUrlDialog(modelName);

                if (downloadUrl) {
                    await this.handleNewDownloadUrl(modelName, downloadUrl, data);
                }
            }
        }
    }

    /**
     * Handle a new download URL
     * @private
     * @param {string} modelName - Name of the model
     * @param {string} downloadUrl - URL to download from
     * @param {Object} statusData - Current status data
     */
    async handleNewDownloadUrl(modelName, downloadUrl, statusData) {
        try {
            const response = await api.fetchApi('/uiapi/add_model_url', {
                method: 'POST',
                body: JSON.stringify({
                    model_name: modelName,
                    url: downloadUrl
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to add model URL: ${response.status}`);
            }

            const urlData = await response.json();
            if (urlData.status === 'ok') {
                // Update download table and restart download
                statusData.download_table = statusData.download_table || {};
                statusData.download_table[modelName] = urlData.model_def;

                await this.downloadModels(statusData.download_table);
            }
        } catch (err) {
            console.error("[DownloadManager] Error handling new download URL:", err);
        }
    }

    /**
     * Stop checking a specific download task
     * @param {string} taskId - ID of the task to stop checking
     */
    stopChecking(taskId) {
        this.activeChecks.delete(taskId);
    }

    /**
     * Stop all active status checks
     */
    stopAllChecks() {
        this.activeChecks.clear();
    }
}

// Create and export a singleton instance
export const downloadManager = new DownloadManager(); 