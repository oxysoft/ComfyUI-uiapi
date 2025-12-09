import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { getNodeDataByPath } from "../utils/nodeUtils.js";
import { getNodes } from "../utils/nodeUtils.js";
import { showBatchDownloadUrlDialog } from "../components/BatchDownloadDialog.js";
import { showWorkflowDialog } from "../components/WorkflowDialog.js";

// Client ID management
let clientId = "-1"; // Initial state before server assigns ID

export function setClientId(id) {
    clientId = id;
    console.log("[ApiHandlers] Client ID set to:", id);
}

export function getClientId() {
    return clientId;
}

/**
 * Safe wrapper for async handlers with error handling
 * @param {Function} handler - Async handler function
 * @param {CustomEvent} event - Event object
 */
async function safeHandleRequest(handler, event) {
    const { request_id } = event.detail;
    try {
        await handler(event);
    } catch (error) {
        console.error(`[ApiHandlers] Error in handler:`, error);
        try {
            // Attempt to send error response back to server
            await postResponse({
                error: true,
                message: error.message || String(error)
            }, request_id);
        } catch (responseError) {
            console.error("[ApiHandlers] Failed to send error response:", responseError);
        }
    }
}

/**
 * Post a response to the API
 * @param {Object} object - Response object
 * @param {string} requestId - Request ID
 */
async function postResponse(object, requestId) {
    const response = await api.fetchApi("/uiapi/webui_response", {
        method: "POST",
        body: JSON.stringify({
            response: object ?? {},
            request_id: requestId,
            client_id: clientId
        })
    });

    if (!response.ok) {
        const errorText = await response.text().catch(() => "Unable to read error response");
        console.error("[ApiHandlers] Failed to post response:", response.status, errorText);
        throw new Error(`Failed to post response: ${response.status} ${errorText}`);
    }

    console.log("[ApiHandlers] Posted response", object ?? {}, "request_id:", requestId, "client_id:", clientId);
}

/**
 * Handle workflow retrieval request
 * @param {CustomEvent} event - Event object
 */
export async function handleGetWorkflow(event) {
    await handleRequest(event);
    const { request_id } = event.detail;
    const prompt = await app.graphToPrompt();
    await postResponse({ workflow: prompt }, request_id);
}

/**
 * Handle workflow API retrieval request
 * @param {CustomEvent} event - Event object
 */
export async function handleGetWorkflowApi(event) {
    await handleRequest(event);
    const { request_id } = event.detail;
    const prompt = await app.graphToPrompt();
    await postResponse({ output: prompt.output }, request_id);
}

/**
 * Handle field retrieval request
 * @param {CustomEvent} event - Event object
 */
export async function handleGetFields(event) {
    await handleRequest(event);
    const { fields, verbose, request_id } = event.detail;

    if (verbose) {
        console.log(`[ApiHandlers] Getting fields: ${JSON.stringify(fields)}`);
    }

    const results = {};

    for (const path of fields) {
        const { node, widget } = getNodeDataByPath(path);
        if (widget) {
            results[path] = widget.value;
        } else {
            console.warn(`[ApiHandlers] Widget not found for path: ${path}`);
            console.log('[ApiHandlers] Available nodes:', node.widgets.map(w => w.name));
            results[path] = null;
        }
    }

    if (verbose) {
        console.log('[ApiHandlers] Retrieved values:', results);
    }

    await postResponse(results, request_id);
}

/**
 * Handle field setting request
 * @param {CustomEvent} event - Event object
 */
export async function handleSetFields(event) {
    await handleRequest(event);
    const { field, fields, verbose, request_id } = event.detail;
    const fieldsToProcess = fields || [field];

    if (verbose) {
        console.log(`[ApiHandlers] Setting fields: ${JSON.stringify(fieldsToProcess)}`);
    }

    for (const [path, value] of fieldsToProcess) {
        const { widget } = getNodeDataByPath(path);
        if (widget) {
            widget.value = value;
        }
    }

    await postResponse(null, request_id);
}

/**
 * Handle connection setting request
 * @param {CustomEvent} event - Event object
 */
export async function handleSetConnection(event) {
    await handleRequest(event);
    const { field: [path1, path2], verbose, request_id } = event.detail;

    if (verbose) {
        console.log(`[ApiHandlers] Setting connection: ${path1} -> ${path2}`);
    }

    const node1 = getNodeDataByPath(path1, true);
    const node2 = getNodeDataByPath(path2, true);

    if (node2.inputs?.length > 0 && node1.outputs?.length > 0) {
        node1.node.connect(node1.outputs[0], node2.node, node2.inputs[0]);
    }

    await postResponse(null, request_id);
}

export async function handleRequest(event) {
    const { client_id, request_id, endpoint, data } = event.detail;
    
    if (client_id !== clientId) {
        setClientId(client_id);
    }
}

/**
 * Handle workflow execution request
 * @param {CustomEvent} event - Event object
 */
export async function handleExecute(event) {
    await handleRequest(event);
    const { request_id } = event.detail;
    const prompt = await app.graphToPrompt();
    const res = await api.queuePrompt(0, prompt);
    await postResponse(res, request_id);
}

/**
 * Handle field query request
 * @param {CustomEvent} event - Event object
 */
export async function handleQueryFields(event) {
    await handleRequest(event);
    const { verbose, request_id } = event.detail;
    
    const nodes = getNodes().map(node => ({
        id: node.id,
        type: node.type,
        title: node.title,
        inputs: node.inputs ? node.inputs.map(input => input.name) : [],
        outputs: node.outputs ? node.outputs.map(output => output.name) : []
    }));

    if (verbose) {
        console.log('[ApiHandlers] Node data gathered:', nodes);
    }

    await postResponse({ nodes }, request_id);
}

/**
 * Handle model URL request
 * @param {CustomEvent} event - Event object
 */
export async function handleGetModelUrl(event) {
    await handleRequest(event);
    console.log("[ApiHandlers] handleGetModelUrl");
    const { ckpt_name, requested_ckpts, request_id } = event.detail;
    
    // If we have multiple models, use batch dialog
    if (requested_ckpts && requested_ckpts.length > 0) {
        // Filter out models that already exist
        const missingCkpts = requested_ckpts.filter(ckpt => !event.detail.existing_ckpts?.includes(ckpt));
        if (missingCkpts.length === 0) {
            await postResponse({}, request_id);
            return;
        }
        const urlMap = await showBatchDownloadUrlDialog(missingCkpts);
        await postResponse(urlMap ? Object.fromEntries(urlMap) : null, request_id);
        return;
    }
    
    // Single model fallback
    const urlMap = await showBatchDownloadUrlDialog([ckpt_name]);
    if (urlMap && urlMap.size > 0) {
        const url = urlMap.get(ckpt_name);
        await postResponse(url ? { url } : null, request_id);
    } else {
        await postResponse(null, request_id);
    }
}

/**
 * Handle workflow dialog request
 * @param {CustomEvent} event - Event object
 */
export async function handleShowWorkflowDialog(event) {
    await handleRequest(event);
    const { workflow, message, title, request_id } = event.detail;

    try {
        // Show dialog and get user's choice
        const accepted = await showWorkflowDialog(workflow, message, title);

        // If accepted, load the workflow
        if (accepted) {
            try {
                // Clear current graph
                app.graph.clear();
                
                // Load new workflow
                app.loadGraphData(workflow);
                
                console.log("[ApiHandlers] Workflow loaded successfully");
            } catch (error) {
                console.error("[ApiHandlers] Error loading workflow:", error);
                await postResponse({
                    accepted: false,
                    message: "Error loading workflow: " + error.message
                }, request_id);
                return;
            }
        }

        // Post response
        await postResponse({
            accepted,
            message: accepted ? "Workflow loaded" : "Workflow rejected"
        }, request_id);
    } catch (error) {
        console.error("[ApiHandlers] Error showing workflow dialog:", error);
        await postResponse({
            accepted: false,
            message: "Error showing workflow dialog: " + error.message
        }, request_id);
    }
}

/**
 * Handle load workflow request (no confirmation dialog)
 * @param {CustomEvent} event - Event object
 */
export async function handleLoadWorkflow(event) {
    await handleRequest(event);
    const { workflow, request_id } = event.detail;

    try {
        // Clear current graph
        app.graph.clear();

        // Load new workflow - ComfyUI handles API format and auto-layout
        await app.loadGraphData(workflow);

        console.log("[ApiHandlers] Workflow loaded successfully");
        await postResponse({
            loaded: true,
            message: "Workflow loaded"
        }, request_id);
    } catch (error) {
        console.error("[ApiHandlers] Error loading workflow:", error);
        await postResponse({
            loaded: false,
            message: "Error loading workflow: " + error.message
        }, request_id);
    }
}

/**
 * Register all API event handlers
 */
export function registerApiHandlers() {
    console.log("[ApiHandlers] registerApiHandlers")
    // Wrap all handlers with error handling
    api.addEventListener("/uiapi/get_workflow", (e) => safeHandleRequest(handleGetWorkflow, e));
    api.addEventListener("/uiapi/get_workflow_api", (e) => safeHandleRequest(handleGetWorkflowApi, e));
    api.addEventListener("/uiapi/get_field", (e) => safeHandleRequest(handleGetFields, e));
    api.addEventListener("/uiapi/get_fields", (e) => safeHandleRequest(handleGetFields, e));
    api.addEventListener("/uiapi/set_fields", (e) => safeHandleRequest(handleSetFields, e));
    api.addEventListener("/uiapi/set_connection", (e) => safeHandleRequest(handleSetConnection, e));
    api.addEventListener("/uiapi/execute", (e) => safeHandleRequest(handleExecute, e));
    api.addEventListener("/uiapi/query_fields", (e) => safeHandleRequest(handleQueryFields, e));
    api.addEventListener("/uiapi/get_model_url", (e) => safeHandleRequest(handleGetModelUrl, e));
    api.addEventListener("/uiapi/show_workflow_dialog", (e) => safeHandleRequest(handleShowWorkflowDialog, e));
    api.addEventListener("/uiapi/load_workflow", (e) => safeHandleRequest(handleLoadWorkflow, e));

    console.log("[ApiHandlers] ✓ All handlers registered");
} 