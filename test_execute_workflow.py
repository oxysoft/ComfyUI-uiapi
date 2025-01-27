import asyncio
import json
import os
from pathlib import Path
import random
import sys
from PIL import Image
import io

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from comfy_client import ComfyClient

async def main():
    # Initialize client
    client = ComfyClient("127.0.0.1:8188")
    
    # Load workflow from test_workflow.json
    workflow_path = Path(__file__).parent / "test_workflow_api.json"
    if not workflow_path.exists():
        print(f"Error: {workflow_path} not found")
        return
        
    with open(workflow_path) as f:
        workflow = json.load(f)
        
    print(f"Workflow: {workflow}")
    print("\nExecuting workflow...")
    
    try:
        # Execute workflow and wait for results
        result = await client.execute_workflow_async(workflow, fields={"KSampler.seed": random.randint(0, 1000000)})
        
        # Process and save outputs
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        for node_id, images in result["outputs"].items():
            for idx, img_np in enumerate(images):
                # Save image
                output_path = output_dir / f"output_{node_id}_{idx}.png"
                image = Image.fromarray(img_np)
                image.save(output_path)
                print(f"Saved output image: {output_path}")
                
        print("\nWorkflow execution completed!")
        print(f"Prompt ID: {result['prompt_id']}")
        print(f"Output directory: {output_dir}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    print("Testing workflow execution...")
    print("Make sure ComfyUI is running and test_workflow.json exists in the same directory")
    asyncio.run(main()) 