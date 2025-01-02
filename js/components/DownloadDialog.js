/**
 * Class representing a modal dialog for model downloads
 */
export class DownloadDialog {
    /**
     * Create a new download dialog
     * @param {string} modelName - Name of the model to download
     */
    constructor(modelName) {
        this.modelName = modelName;
        this.modal = null;
        this.content = null;
        this.input = null;
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
            min-width: 400px;
            color: #ffffff;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
        `;

        // Create title
        const title = document.createElement('h3');
        title.textContent = `Enter Download URL for ${this.modelName}`;
        title.style.cssText = `
            margin: 0 0 15px 0;
            color: #ffffff;
            font-size: 16px;
        `;

        // Create input field
        this.input = document.createElement('input');
        this.input.type = 'text';
        this.input.placeholder = 'Enter model download URL';
        this.input.style.cssText = `
            width: 100%;
            padding: 8px;
            margin-bottom: 15px;
            background: #1a1a1a;
            border: 1px solid #3d3d3d;
            color: #ffffff;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 14px;
        `;

        // Create buttons container
        const buttons = document.createElement('div');
        buttons.style.cssText = `
            display: flex;
            justify-content: flex-end;
            gap: 10px;
        `;

        // Add elements to the DOM
        buttons.appendChild(this.createButton('Cancel', '#e74c3c'));
        buttons.appendChild(this.createButton('Submit', '#2ecc71'));
        
        this.content.appendChild(title);
        this.content.appendChild(this.input);
        this.content.appendChild(buttons);
        this.modal.appendChild(this.content);

        // Focus the input field
        setTimeout(() => this.input.focus(), 0);
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
     * Show the dialog and return a promise that resolves with the URL
     * @returns {Promise<string|null>} The entered URL or null if cancelled
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
                const url = this.input.value.trim();
                cleanup();
                resolve(url || null);
            };

            // Handle keyboard events
            this.input.onkeydown = (e) => {
                if (e.key === 'Enter') {
                    buttons[1].click();
                } else if (e.key === 'Escape') {
                    buttons[0].click();
                }
            };

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
 * Show a download URL dialog
 * @param {string} modelName - Name of the model
 * @returns {Promise<string|null>} The entered URL or null if cancelled
 */
export async function showDownloadUrlDialog(modelName) {
    const dialog = new DownloadDialog(modelName);
    return await dialog.show();
} 