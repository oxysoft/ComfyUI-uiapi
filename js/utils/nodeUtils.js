import { app } from "../../../scripts/app.js";
import { PRIMARY_WIDGETS } from "../config/constants.js";

/**
 * Get all nodes in the graph, ensuring they have titles
 * @returns {Array} Array of nodes
 */
export function getNodes() {
    app.graph._nodes.forEach(node => {
        if (!node.title) node.title = "";
    });
    return app.graph._nodes;
}

/**
 * Calculate distance between two points
 * @param {number} x1 - First point x coordinate
 * @param {number} y1 - First point y coordinate
 * @param {number} x2 - Second point x coordinate
 * @param {number} y2 - Second point y coordinate
 * @returns {number} Distance between points
 */
export const distance = (x1, y1, x2, y2) => Math.hypot(x2 - x1, y2 - y1);

/**
 * Connect two nodes together
 * @param {string} path1 - Path to first node
 * @param {string} path2 - Path to second node
 */
export function connectNodes(path1, path2) {
    const node1 = getNodeDataByPath(path1, true);
    const node2 = getNodeDataByPath(path2, true);

    const outputNode = node1.node;
    const outputs = node1.outputs;
    const inputNode = node2.node;
    const inputs = node2.inputs;

    if (node2.inputs?.length > 0 && node1.outputs?.length > 0) {
        outputNode.connect(outputs[0], inputNode, inputs[0]);
    }
}

/**
 * Select a node at specific coordinates
 * @param {number} x - X coordinate
 * @param {number} y - Y coordinate
 */
export function selectNodeAt(x, y) {
    const node = app.graph.getNodeOnPos(x, y, app.canvas.visible_nodes, 5);
    if (!node) return;

    app.canvas.deselectAllNodes();
    app.canvas.selectNodes([node]);
}

/**
 * Get node data by search path
 * @param {string} searchPath - Path to search for
 * @param {boolean} slotsAsIndices - Whether to return slot indices instead of objects
 * @returns {Object} Node data object
 */
export function getNodeDataByPath(searchPath, slotsAsIndices = false) {
    function getDefaultWidget(node) {
        const widgetName = PRIMARY_WIDGETS[node.type];
        return widgetName ? node.widgets.find(w => w.name === widgetName) : null;
    }

    function findMatchingComponents(node, componentName) {
        const indexMatch = componentName.match(/^(inputs|outputs)\[(\d+)\]$/);
        if (indexMatch) {
            const [, ioType, idx] = indexMatch;
            const index = parseInt(idx);
            return {
                [ioType]: node[ioType][index] ? [slotsAsIndices ? index : node[ioType][index]] : []
            };
        }

        const findComponents = (array, predicate) =>
            array?.reduce((acc, item, index) => {
                if (predicate(item)) {
                    acc.push(slotsAsIndices ? index : item);
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
    const result = { node: null, widget: null, inputs: [], outputs: [] };

    for (const node of getNodes()) {
        const nodeTitle = node.title.toLowerCase();
        const nodeType = node.type.toLowerCase();

        if (nodeTitle === primary || nodeType === primary) {
            result.node = node;

            if (secondary) {
                const { widget, inputs, outputs } = findMatchingComponents(node, secondary);
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
            const { widget, inputs, outputs } = findMatchingComponents(node, primary);
            if (widget || inputs.length || outputs.length) {
                result.node = node;
                if (widget) result.widget = widget;
                result.inputs = inputs;
                result.outputs = outputs;
                break;
            }
        }
    }

    return result;
}

/**
 * Simulate a click event at specific coordinates
 * @param {number} x - X coordinate
 * @param {number} y - Y coordinate
 */
export function simulateClick(x, y) {
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