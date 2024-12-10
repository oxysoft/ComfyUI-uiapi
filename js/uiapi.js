import {app} from "../../scripts/app.js"
import {api} from "../../scripts/api.js"

let mx = 0
let my = 0

let cx = 0
let cy = 0

let graph = null
let canvas = null
let c = null
let ctxmenu = null

let last_rename_value = ""
let last_click_time = null


const rename_defaults = [
    'prompt', 'promptneg',
    'cfg', 'chg', 'img',
    'ccg{ControlNetApply}', 'cn_img{LoadImage}'
]

const primaryWidgets = {
    CLIPSetLastLayer: "stop_at_clip_layer",
    CLIPTextEncode: "text",
    VAELoader: "vae_name",
    TomePatchModel: "ratio",
    SaveImage: "filename_prefix",
    LoadImage: "image"
};


console.log("=== uiapi.js ===")

const colors = {
    loader: [0, 0.4, 0.3],
    clip: [20, 0.4, 0.3],
    note: [40, 0.4, 0.3],
    sampler: [60, 0.4, 0.3],
    controlnet: [80, 0.4, 0.3],
    vae: [100, 0.4, 0.3],
    conditioning: [120, 0.4, 0.3],
    latent: [140, 0.4, 0.3],
    mask: [160, 0.4, 0.3],
    image: [180, 0.4, 0.3],
    style: [200, 0.4, 0.3],
    primitive: [220, 0.4, 0.3],
    gligen: [240, 0.4, 0.3],
};

app.registerExtension({
    name: "uiapi",
    version: "1.0",
    setup: function () {
        console.log("==== uiapi setup ====")
        graph = app.graph
        canvas = c = app.canvas
        graph.searchNodes = getNodeDataByPath;
        graph.connectNodes = connectNodes;

        api.addEventListener("/uiapi/get_workflow", get_workflow)
        api.addEventListener("/uiapi/get_field", getFieldsHandler)
        api.addEventListener("/uiapi/get_fields", getFieldsHandler)
        api.addEventListener("/uiapi/set_fields", setFieldsHandler)
        api.addEventListener("/uiapi/set_connection", setConnectionHandler)
        api.addEventListener("/uiapi/execute", executeHandler)
        api.addEventListener("/uiapi/query_fields", queryFieldsHandler)

        let onMouseDown = canvas.onMouseDown
        canvas.onMouseDown = e => {
            onMouse(e)
            if (onMouseDown)
                onMouseDown(e)
        }

        let onMouseUp = canvas.onMouseUp
        canvas.onMouseUp = e => {
            onMouse(e)
            if (onMouseUp)
                onMouseUp(e)
        }

        let onMouseMove = canvas.onMouseMove
        canvas.onMouseMove = e => {
            onMouse(e)
            if (onMouseMove)
                onMouseMove(e)
        }

        window.addEventListener("keydown", onKey)
        window.addEventListener("keyup", onKey)
    },
})

function onMouse(e) {
    // mx = c.graph_mouse[0]
    // my = c.graph_mouse[1]

    console.log(e.type)
    switch (e.type) {
        case "mousedown":
        case "pointerdown":
            // ctxmenu?.close()
            // ctxmenu = null

            // if (Date.now() - last_click_time < 200) {
            //     c.search_box?.close()
            // }
            break
    }


    last_click_time = Date.now()
}

function onKey(e) {
    let block = false;
    let c = app.canvas

    if (e.type === "keydown" && !e.repeat) {
        if (e.key == 'escape') {
            ctxmenu?.close()
            ctxmenu = null

            c.search_box?.close()
        }

        let movedist = distance(cx, cy, c.mouse[0], c.mouse[1])
        let auto = movedist < 10

        // F2 rename feature
        if (e.key === 'F2') {
            if (ctxmenu != null && auto) {
                // Auto-select next numbered rename_default if last_rename_value ends with a number
                let last_char = last_rename_value.slice(-1)
                let last_num = parseInt(last_char)
                let next_num = (last_num + 1) % 10
                let next_value = last_rename_value.slice(0, -1) + next_num
                if (getRenameDefaults().includes(next_value)) {
                    for (let node of Object.values(app.canvas.selected_nodes))
                        node.title = next_value
                    last_rename_value = next_value
                    ctxmenu.close()
                    ctxmenu = null
                }
            } else {
                selectNodeAt(c.graph_mouse[0], c.graph_mouse[1])
                showRenameContextMenu(e)
                // // Ask user to input new title
                // let new_title = prompt("Enter new title", Object.values(c.selected_nodes)[0].title)
                // if (new_title) {
                //     for (let node of Object.values(app.canvas.selected_nodes)) {
                //         node.title = new_title
                //     }
                // }
            }
            block = true
        }
    }

    this.graph.change()

    if (block) {
        e.preventDefault()
        e.stopImmediatePropagation()
        return false
    }
}


function getRenameDefaults() {
    let ret = []

    // Add the defaults
    for (let rename_entry of rename_defaults) {
        // Check for {nodeType}, slice the nodeType inside using regex, count the number of nodeType nodes in the graph, and add this many numbered entries
        if (rename_entry.includes("{")) {
            let nodeType = rename_entry.match(/\{(.*)\}/)[1]
            let count = app.graph._nodes.filter(n => n.type === nodeType).length
            for (let i = 0; i < count; i++) {
                ret.push(rename_entry.replace(`{${nodeType}}`, i))
            }
        } else {
            ret.push(rename_entry)
        }
    }

    return ret
}

function showRenameContextMenu(e) {
    cx = c.mouse[0]
    cy = c.mouse[1]

    ctxmenu?.close()
    ctxmenu = new LiteGraph.ContextMenu(getRenameDefaults(), {
        title: `Rename '${Object.values(c.selected_nodes)[0].title}' ...`,
        left: cx - 30,
        top: cy - 30,
        callback: (v, options, e) => {
            for (let node of Object.values(app.canvas.selected_nodes)) {
                node.title = v
            }
            last_rename_value = v
        }
    });


    return false;
}


//region API Handlers
async function get_workflow(ev) {
    let prompt = await app.graphToPrompt()

    await api.fetchApi("/uiapi/response", {
        method: "POST",
        body: JSON.stringify({
            "workflow": prompt
        })
    })
}

async function post_response(object) {
    await api.fetchApi("/uiapi/response", {
        method: "POST",
        body: JSON.stringify(object ?? {})
    })
    console.log("post_response", object ?? {})
}

/**
 * Searches for a node and its components based on a search path.
 * @param {string} searchPath - The search path in format "primary" or "primary.secondary".
 * @param {boolean} [slots_as_indices=false] - If true, return indices for inputs and outputs instead of objects.
 * @returns {Object} An object containing the matched node, widget, inputs, and outputs (or their indices).
 */
function getNodeDataByPath(searchPath, slots_as_indices = false) {
    function getDefaultWidget(node) {
        const widgetName = primaryWidgets[node.type];
        return widgetName ? node.widgets.find(w => w.name === widgetName) : null;
    }

    function findMatchingComponents(node, componentName) {
        const indexMatch = componentName.match(/^(inputs|outputs)\[(\d+)\]$/);
        if (indexMatch) {
            const [, ioType, idx] = indexMatch;
            const index = parseInt(idx);
            return {
                [ioType]: node[ioType][index] ? [slots_as_indices ? index : node[ioType][index]] : []
            };
        }

        const findComponents = (array, predicate) =>
            array?.reduce((acc, item, index) => {
                if (predicate(item)) {
                    acc.push(slots_as_indices ? index : item);
                }
                return acc;
            }, []) || [];

        return {
            widget: node.widgets?.find(w => w.name.toLowerCase() === componentName),
            inputs: findComponents(node.inputs, i => i.name.toLowerCase() === componentName),
            outputs: findComponents(node.outputs, o => o.name.toLowerCase() === componentName)
        };
    }

    const [primary, secondary] = searchPath.toLowerCase().split('.');
    let result = {node: null, widget: null, inputs: [], outputs: []};

    for (const node of app.graph._nodes) {
        const nodeTitle = node.title.toLowerCase();
        const nodeType = node.type.toLowerCase();

        if (nodeTitle === primary || nodeType === primary) {
            result.node = node;

            if (secondary) {
                const {
                    widget,
                    inputs,
                    outputs
                } = findMatchingComponents(node, secondary);
                if (widget) result.widget = widget;
                if (inputs) result.inputs = inputs;
                if (outputs) result.outputs = outputs;
            } else {
                const defaultWidget = getDefaultWidget(node);
                if (defaultWidget) result.widget = defaultWidget;
            }

            if (result.widget || result.inputs.length || result.outputs.length) {
                break;
            }
        } else if (!secondary) {
            const {
                widget,
                inputs,
                outputs
            } = findMatchingComponents(node, primary);
            if (widget || inputs.length || outputs.length) {
                result.node = node;
                if (widget) result.widget = widget;
                result.inputs = inputs;
                result.outputs = outputs;
                break;
            }
        }
    }

    console.log(`getNodeDataByPath(${searchPath}, slots_as_indices=${slots_as_indices}) ->`, result);
    return result;
}

/**
 * Handle the getFields event, where the incoming request
 * retrieves values from specified node fields.
 * @param {CustomEvent} ev - The event object containing the fields to retrieve.
 * @returns {Promise<Object>}
 */
async function getFieldsHandler(ev) {
    const { fields, verbose } = ev.detail;

    if (verbose) {
        console.log(`getFieldsHandler(${JSON.stringify(fields)})`);
    }

    const results = {};

    for (const path of fields) {
        const { node, widget } = getNodeDataByPath(path);
        if (widget) {
            results[path] = widget.value;
        } else {
            console.warn(`Widget not found for path: ${path}`);
            console.log('Available nodes:', node.widgets.map(w => w.name));
            results[path] = null;
        }
    }

    if (verbose) {
        console.log('Retrieved values:', results);
    }

    await post_response(results);
    return results;
}

/**
 * Handle the setField event, where the incoming request
 * allows to set single or multiple field values in nodes.
 * @param {CustomEvent} ev - The event object containing the field(s) data.
 * @returns {Promise<void>}
 */
async function setFieldsHandler(ev) {
    const { field, fields, verbose } = ev.detail;

    let fieldsToProcess = fields || [field];

    if (verbose) {
        console.log(`setFieldHandler(${JSON.stringify(fieldsToProcess)})`);
        logGraphDetails(app.graph._nodes);
    }

    console.log("fieldsToProcess", fieldsToProcess)
    for (const [path, value] of fieldsToProcess) {
        const { widget } = getNodeDataByPath(path);
        console.log("[path, value] = ", path, value)
        if (widget) {
            widget.value = value;
        }
    }

    await post_response();
}

/**
 * Handle the setConnection event, where the incoming request
 * allows to connect two nodes.
 * @param {CustomEvent} ev - The event object containing the connection data.
 * @returns {Promise<void>}
 */
async function setConnectionHandler(ev) {
    const {field: [path1, path2], verbose} = ev.detail;

    if (verbose) {
        console.log(`setConnectionHandler(${path1} -> ${path2})`);
        logGraphDetails(app.graph._nodes);
    }

    connectNodes(path1, path2);

    await post_response();
}

/**
 * Log details of all nodes in the graph.
 * @param {Array} nodes - The array of nodes in the graph.
 */
function logGraphDetails(nodes) {
    nodes.forEach(node => {
        console.log(`  node: ${node.title} (${node.type})`);
        if (node.widgets) {
            node.widgets.forEach(widget => {
                console.log(`    widget: ${widget.name} (${widget.value})`);
            });
        }
    });
}

async function executeHandler(e) {
    let prompt = await app.graphToPrompt()

    const res = await api.queuePrompt(-1, prompt)
    await post_response(res)
}

async function queryFieldsHandler(e) {
    const { verbose } = e.detail;
    
    // Gather an array of all nodes with their input and output names
    const nodes = app.graph._nodes.map(node => ({
        id: node.id,
        type: node.type,
        title: node.title,
        inputs: node.inputs ? node.inputs.map(input => input.name) : [],
        outputs: node.outputs ? node.outputs.map(output => output.name) : []
    }));

    if (verbose) {
        console.log('queryFieldsHandler: Node data gathered -', nodes);
    }

    // Post the gathered data to the server
    const results = await post_response({ 
        "nodes": nodes
     });

    if (verbose) {
        console.log('queryFieldsHandler: Data posted to server -', results);
    }
}

//region Utilities
const distance = (x1, y1, x2, y2) => Math.hypot(x2 - x1, y2 - y1);

function connectNodes(path1, path2) {
    let node1 = getNodeDataByPath(path1, true);
    let node2 = getNodeDataByPath(path2, true);

    let output_node = node1.node;
    let outputs = node1.outputs;
    let input_node = node2.node;
    let inputs = node2.inputs;


    if (node2.inputs?.length > 0 && node1.outputs?.length > 0) {
        output_node.connect(outputs[0], input_node, inputs[0]);
    }
}

function selectNodeAt(x, y) {
    const node = app.graph.getNodeOnPos(x, y, canvas.visible_nodes, 5);
    if (node == null)
        return;

    app.canvas.deselectAllNodes();
    app.canvas.selectNodes([node]);
}

function click(x, y) {
    function click(x, y) {
        const el = document.elementFromPoint(x, y);
        const ev = new MouseEvent("click", {
            view: window,
            bubbles: true,
            cancelable: true,
            clientX: x,
            clientY: y
        });

        el.dispatchEvent(ev);
    }
}

//endregion

function uncolor(app) {
    app.graph._nodes.forEach((node) => {
        if (node.type.toLowerCase() === "note") {
            const [h, s, l] = colors.note;
            const bgcolor = hslToHex(h / 360, s, l);
            node.bgcolor = bgcolor;
            node.color = shadeHexColor(node.bgcolor);
        } else {
            node.bgcolor = hslToHex(0, 0, 0.3);
            node.color = shadeHexColor(node.bgcolor);
        }
        node.setDirtyCanvas(true, true);
    });
}

function hslToRgb(h, s, l) {
    let r, g, b;

    if (s === 0) {
        r = g = b = l; // Achromatic
    } else {
        const hue2rgb = function hue2rgb(p, q, t) {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };

        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
    }

    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function rgbToHex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

function hslToHex(h, s, l) {
    const [r, g, b] = hslToRgb(h, s, l);
    return rgbToHex(r, g, b);
}

function shadeHexColor(hex, amount = -0.2) {
    // Remove the # symbol if it exists
    if (hex.startsWith("#")) {
        hex = hex.slice(1);
    }

    // Convert the hex values to decimal (base 10) integers
    let r = parseInt(hex.slice(0, 2), 16);
    let g = parseInt(hex.slice(2, 4), 16);
    let b = parseIGnt(hex.slice(4, 6), 16);

    // Apply the shade amount to each RGB component
    r = Math.max(0, Math.min(255, r + amount * 100));
    g = Math.max(0, Math.min(255, g + amount * 100));
    b = Math.max(0, Math.min(255, b + amount * 100));

    // Convert the updated RGB values back to HEX
    return rgbToHex(r, g, b);
}

function getContrastColor(hexColor) {
    // Remove the # symbol if it exists
    if (hexColor.startsWith("#")) {
        hexColor = hexColor.slice(1);
    }

    // Convert the hex values to decimal (base 10) integers
    const r = parseInt(hexColor.slice(0, 2), 16) / 255;
    const g = parseInt(hexColor.slice(2, 4), 16) / 255;
    const b = parseInt(hexColor.slice(4, 6), 16) / 255;

    // Calculate the relative luminance
    const L = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    // Use the contrast ratio to determine the text color
    return L > 0.179 ? "#000000" : "#FFFFFF";
}

// LiteGraph.search_hide_on_mouse_leave = false
