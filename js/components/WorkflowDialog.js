/**
 * Class representing a modal dialog for workflow approval
 */
export class WorkflowDialog {
    /**
     * Create a new workflow approval dialog
     * @param {Object} workflow - The workflow data to display
     * @param {string} message - Message to show to the user
     * @param {string} title - Dialog title
     */
    constructor(workflow, message, title) {
        this.workflow = workflow;
        this.message = message;
        this.title = title;
        this.modal = null;
        this.content = null;
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
        title.textContent = this.title;
        title.style.cssText = `
            margin: 0 0 15px 0;
            font-size: 1.2em;
            color: #ffffff;
        `;

        // Create message
        const message = document.createElement('p');
        message.textContent = this.message;
        message.style.cssText = `
            margin: 0 0 20px 0;
            color: #cccccc;
        `;

        // Create workflow preview (optional)
        const preview = document.createElement('div');
        preview.style.cssText = `
            margin: 0 0 20px 0;
            padding: 10px;
            background: #1d1d1d;
            border-radius: 4px;
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.9em;
            color: #aaaaaa;
        `;
        preview.textContent = JSON.stringify(this.workflow, null, 2);

        // Create buttons container
        const buttons = document.createElement('div');
        buttons.style.cssText = `
            display: flex;
            justify-content: flex-end;
            gap: 10px;
        `;

        // Add buttons
        const acceptButton = this.createButton('Accept', '#2ecc71');
        const rejectButton = this.createButton('Reject', '#e74c3c');
        
        buttons.appendChild(rejectButton);
        buttons.appendChild(acceptButton);

        // Assemble modal
        this.content.appendChild(title);
        this.content.appendChild(message);
        this.content.appendChild(preview);
        this.content.appendChild(buttons);
        this.modal.appendChild(this.content);
    }

    /**
     * Create a styled button
     * @private
     */
    createButton(text, bgColor) {
        const button = document.createElement('button');
        button.textContent = text;
        button.style.cssText = `
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: ${bgColor};
            color: white;
            cursor: pointer;
            font-size: 1em;
            transition: opacity 0.2s;
        `;
        button.onmouseover = () => button.style.opacity = '0.8';
        button.onmouseout = () => button.style.opacity = '1';
        return button;
    }

    /**
     * Show the dialog and return a promise that resolves with the user's choice
     * @returns {Promise<boolean>} True if accepted, false if rejected
     */
    show() {
        return new Promise((resolve) => {
            this.createModal();
            document.body.appendChild(this.modal);

            // Handle button clicks
            const buttons = this.content.querySelectorAll('button');
            buttons[0].onclick = () => {  // Reject
                this.modal.remove();
                resolve(false);
            };
            buttons[1].onclick = () => {  // Accept
                this.modal.remove();
                resolve(true);
            };

            // Handle escape key
            const handleEscape = (e) => {
                if (e.key === 'Escape') {
                    this.modal.remove();
                    resolve(false);
                    document.removeEventListener('keydown', handleEscape);
                }
            };
            document.addEventListener('keydown', handleEscape);
        });
    }
}

/**
 * Show a workflow approval dialog
 * @param {Object} workflow - The workflow data to display
 * @param {string} message - Message to show to the user
 * @param {string} title - Dialog title
 * @returns {Promise<boolean>} True if accepted, false if rejected
 */
export async function showWorkflowDialog(workflow, message, title) {
    const dialog = new WorkflowDialog(workflow, message, title);
    return await dialog.show();
} 