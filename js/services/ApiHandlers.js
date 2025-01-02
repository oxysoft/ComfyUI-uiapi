import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { getNodeDataByPath } from "../utils/nodeUtils.js";
import { getNodes } from "../utils/nodeUtils.js";
import { showBatchDownloadUrlDialog } from "../components/BatchDownloadDialog.js";

/**
 * Post a response to the API
 * @param {Object} object - Response object
 * @param {string} requestId - Request ID
 */
async function postResponse(object, requestId) {
    await api.fetchApi("/uiapi/response", {
        method: "POST",
        body: JSON.stringify({
            response: object ?? {},
            request_id: requestId
        })
    });
    console.log("[ApiHandlers] Posted response", object ?? {}, "request_id:", requestId);
}

/**
 * Handle workflow retrieval request
 * @param {CustomEvent} event - Event object
 */
export async function handleGetWorkflow(event) {
    const { request_id } = event.detail;
    const prompt = await app.graphToPrompt();
    await postResponse({ workflow: prompt }, request_id);
}

/**
 * Handle field retrieval request
 * @param {CustomEvent} event - Event object
 */
export async function handleGetFields(event) {
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

/**
 * Handle workflow execution request
 * @param {CustomEvent} event - Event object
 */
export async function handleExecute(event) {
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
    console.log("[ApiHandlers] handleGetModelUrl");
    const { model_name, model_names, request_id } = event.detail;
    
    // If we have multiple models, use batch dialog
    if (model_names && model_names.length > 0) {
        // Filter out models that already exist
        const missingModels = model_names.filter(name => !event.detail.existing_models?.includes(name));
        if (missingModels.length === 0) {
            await postResponse({}, request_id);
            return;
        }
        const urlMap = await showBatchDownloadUrlDialog(missingModels);
        await postResponse(urlMap ? Object.fromEntries(urlMap) : null, request_id);
        return;
    }
    
    // Single model fallback
    const urlMap = await showBatchDownloadUrlDialog([model_name]);
    if (urlMap && urlMap.size > 0) {
        const url = urlMap.get(model_name);
        await postResponse(url ? { url } : null, request_id);
    } else {
        await postResponse(null, request_id);
    }
}

/**
 * Register all API event handlers
 */
export function registerApiHandlers() {
    console.log("[ApiHandlers] registerApiHandlers")
    api.addEventListener("/uiapi/get_workflow", handleGetWorkflow);
    api.addEventListener("/uiapi/get_field", handleGetFields);
    api.addEventListener("/uiapi/get_fields", handleGetFields);
    api.addEventListener("/uiapi/set_fields", handleSetFields);
    api.addEventListener("/uiapi/set_connection", handleSetConnection);
    api.addEventListener("/uiapi/execute", handleExecute);
    api.addEventListener("/uiapi/query_fields", handleQueryFields);
    api.addEventListener("/uiapi/get_model_url", handleGetModelUrl);
} 