import aiohttp
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

async def monitor_download_status(session: aiohttp.ClientSession, task_id: str) -> None:
    """Monitor the status of a download task"""
    while True:
        async with session.get(f'http://127.0.0.1:8188/uiapi/download_status/{task_id}') as response:
            status_data = await response.json()
            
            if status_data.get('status') == 'error':
                print(f"Error: {status_data.get('error')}")
                break
                
            if status_data.get('status') == 'completed':
                print("\nDownload completed!")
                print(json.dumps(status_data.get('progress', {}), indent=2))
                break
                
            progress = status_data.get('progress', {})
            current = status_data.get('current_model', 0)
            total = status_data.get('total_models', 0)
            
            print(f"\rProgress: {current}/{total} models", end='', flush=True)
            
            await asyncio.sleep(1)

async def main():
    # Load workflow from test_workflow.json
    workflow_path = Path(__file__).parent / "test_workflow.json"
    if not workflow_path.exists():
        print(f"Error: {workflow_path} not found")
        return
        
    with open(workflow_path) as f:
        workflow = json.load(f)
    
    # Empty download table to force manual URL input
    download_table = {}
    
    async with aiohttp.ClientSession() as session:
        # Send download request
        print("Sending download request...")
        async with session.post(
            'http://127.0.0.1:8188/uiapi/download_models',
            json={
                'download_table': download_table,
                'workflow': workflow
            }
        ) as response:
            data = await response.json()
            
            if data.get('status') != 'ok':
                print(f"Error: {data.get('error', 'Unknown error')}")
                return
                
            task_id = data.get('download_id')
            print(f"Download task started with ID: {task_id}")
            
            # Monitor the download status
            await monitor_download_status(session, task_id)

if __name__ == "__main__":
    asyncio.run(main()) 