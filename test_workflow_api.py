import aiohttp
import asyncio
import json
from pathlib import Path

async def test_get_workflow_api():
    """Test the get_workflow_api endpoint"""
    print("\nTesting get_workflow_api endpoint...")
    
    async with aiohttp.ClientSession() as session:
        # Call the endpoint
        async with session.post('http://localhost:8188/uiapi/get_workflow_api') as response:
            result = await response.json()
            
            # Pretty print the result
            print("\nResponse:")
            print(json.dumps(result, indent=2))
            
            workflow = result["response"]["output"]
            print(f"Workflow: {workflow}")
            
            # Save to file
            output_path = Path(__file__).parent / "test_workflow_api.json"
            with open(output_path, 'w') as f:
                json.dump(workflow, f, indent=2)
            print(f"\nSaved response to {output_path}")

if __name__ == "__main__":
    asyncio.run(test_get_workflow_api()) 