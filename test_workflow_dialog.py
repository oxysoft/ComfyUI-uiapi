import aiohttp
import asyncio
import json
from pathlib import Path

async def main():
    # Load workflow from test_workflow.json
    workflow_path = Path(__file__).parent / "test_workflow.json"
    if not workflow_path.exists():
        print(f"Error: {workflow_path} not found")
        return
        
    with open(workflow_path) as f:
        workflow = json.load(f)
    
    async with aiohttp.ClientSession() as session:
        # Send workflow dialog request
        print("\nSending workflow for approval...")
        async with session.post(
            'http://127.0.0.1:8188/uiapi/send_workflow',
            json={
                'workflow': workflow,
                'message': "Would you like to load this test workflow?",
                'title': "Test Workflow"
            }
        ) as response:
            data = await response.json()
            
            if data.get('status') == 'error':
                print(f"\nError: {data.get('error', 'Unknown error')}")
                return
                
            if data.get('status') == 'ok':
                accepted = data.get('accepted', False)
                message = data.get('message', '')
                
                if accepted:
                    print("\n✓ Workflow was accepted and loaded!")
                else:
                    print("\n✗ Workflow was rejected")
                    
                if message:
                    print(f"Message: {message}")
            else:
                print(f"\nUnexpected response: {json.dumps(data, indent=2)}")

if __name__ == "__main__":
    print("Testing workflow dialog...")
    print("Make sure ComfyUI is running and test_workflow.json exists in the same directory")
    asyncio.run(main()) 