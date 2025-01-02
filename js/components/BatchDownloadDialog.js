/**
 * Class representing a modal dialog for batch model downloads
 */
export class BatchDownloadDialog {
    /**
     * Create a new batch download dialog
     * @param {string[]} modelNames - Names of the models to download
     */
    constructor(modelNames) {
        this.modelNames = modelNames;
        this.modal = null;
        this.content = null;
        this.inputs = new Map(); // Map of model names to input elements
    }

    /**
     * Create the modal DOM structure
     * @private
     */
    createModal() {
        // Create modal container
        this.modal = document.createElement('div');
        this.modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 99999;
        `;

        // Create modal content
        this.content = document.createElement('div');
        this.content.style.cssText = `
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            min-width: 600px;
            max-width: 80%;
            max-height: 80vh;
            color: #ffffff;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
            overflow-y: auto;
        `;

        // Create title
        const title = document.createElement('h3');
        title.textContent = 'Enter Download URLs for Models';
        title.style.cssText = `
            margin: 0 0 15px 0;
            color: #ffffff;
            font-size: 16px;
        `;

        // Create table container
        const tableContainer = document.createElement('div');
        tableContainer.style.cssText = `
            margin-bottom: 15px;
            max-height: calc(80vh - 120px);
            overflow-y: auto;
        `;

        // Create table
        const table = document.createElement('table');
        table.style.cssText = `
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        `;

        // Create table header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        ['Model Name', 'Download URL'].forEach(text => {
            const th = document.createElement('th');
            th.textContent = text;
            th.style.cssText = `
                text-align: left;
                padding: 8px;
                border-bottom: 2px solid #3d3d3d;
                position: sticky;
                top: 0;
                background: #2d2d2d;
            `;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create table body
        const tbody = document.createElement('tbody');
        this.modelNames.forEach(modelName => {
            const row = document.createElement('tr');
            
            // Model name cell
            const nameCell = document.createElement('td');
            nameCell.textContent = modelName;
            nameCell.style.cssText = `
                padding: 8px;
                border-bottom: 1px solid #3d3d3d;
            `;
            
            // URL input cell
            const urlCell = document.createElement('td');
            urlCell.style.cssText = `
                padding: 8px;
                border-bottom: 1px solid #3d3d3d;
            `;
            
            const input = document.createElement('input');
            input.type = 'text';
            input.placeholder = 'Enter model download URL';
            input.style.cssText = `
                width: 100%;
                padding: 8px;
                background: #1a1a1a;
                border: 1px solid #3d3d3d;
                color: #ffffff;
                border-radius: 4px;
                box-sizing: border-box;
                font-size: 14px;
            `;
            
            this.inputs.set(modelName, input);
            urlCell.appendChild(input);
            
            row.appendChild(nameCell);
            row.appendChild(urlCell);
            tbody.appendChild(row);
        });
        
        table.appendChild(tbody);
        tableContainer.appendChild(table);

        // Create buttons container
        const buttons = document.createElement('div');
        buttons.style.cssText = `
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            position: sticky;
            bottom: 0;
            background: #2d2d2d;
            padding-top: 10px;
        `;

        // Add elements to the DOM
        buttons.appendChild(this.createButton('Cancel', '#e74c3c'));
        buttons.appendChild(this.createButton('Submit All', '#2ecc71'));
        
        this.content.appendChild(title);
        this.content.appendChild(tableContainer);
        this.content.appendChild(buttons);
        this.modal.appendChild(this.content);

        // Focus the first input field
        setTimeout(() => {
            const firstInput = this.inputs.values().next().value;
            if (firstInput) firstInput.focus();
        }, 0);
    }

    /**
     * Create a button element
     * @private
     * @param {string} text - Button text
     * @param {string} bgColor - Background color
     * @returns {HTMLButtonElement} The created button
     */
    createButton(text, bgColor) {
        const button = document.createElement('button');
        button.textContent = text;
        button.style.cssText = `
            padding: 8px 16px;
            border-radius: 4px;
            border: none;
            background: ${bgColor};
            color: white;
            cursor: pointer;
            font-size: 14px;
            transition: opacity 0.2s;
        `;
        button.onmouseover = () => button.style.opacity = '0.8';
        button.onmouseout = () => button.style.opacity = '1';
        return button;
    }

    /**
     * Show the dialog and return a promise that resolves with the URLs
     * @returns {Promise<Map<string,string>|null>} Map of model names to URLs, or null if cancelled
     */
    show() {
        return new Promise((resolve) => {
            this.createModal();
            document.body.appendChild(this.modal);

            const cleanup = () => {
                document.body.removeChild(this.modal);
                this.modal = null;
            };

            // Handle button clicks
            const buttons = this.content.querySelectorAll('button');
            buttons[0].onclick = () => {
                cleanup();
                resolve(null);
            };

            buttons[1].onclick = () => {
                const urlMap = new Map();
                this.inputs.forEach((input, modelName) => {
                    const url = input.value.trim();
                    if (url) {
                        urlMap.set(modelName, url);
                    }
                });
                cleanup();
                resolve(urlMap);
            };

            // Handle keyboard events for each input
            this.inputs.forEach(input => {
                input.onkeydown = (e) => {
                    if (e.key === 'Enter' && e.ctrlKey) {
                        buttons[1].click(); // Submit all on Ctrl+Enter
                    } else if (e.key === 'Escape') {
                        buttons[0].click();
                    }
                };
            });

            // Handle click outside to cancel
            this.modal.onclick = (e) => {
                if (e.target === this.modal) {
                    buttons[0].click();
                }
            };
        });
    }
}

/**
 * Show a batch download URL dialog
 * @param {string[]} modelNames - Names of the models
 * @returns {Promise<Map<string,string>|null>} Map of model names to URLs, or null if cancelled
 */
export async function showBatchDownloadUrlDialog(modelNames) {
    const dialog = new BatchDownloadDialog(modelNames);
    return await dialog.show();
} 