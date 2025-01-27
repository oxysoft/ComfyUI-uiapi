import aiohttp
import asyncio

async def clear_nonexistant_models():
    """Call the clear_nonexistant_models endpoint with a sample workflow"""
    base_url = "http://127.0.0.1:8188"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/uiapi/clear_nonexistant_models"
        ) as response:
            data = await response.json()
            print("Response:", data)

if __name__ == "__main__":
    asyncio.run(clear_nonexistant_models()) 