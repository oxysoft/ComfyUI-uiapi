import asyncio
import logging
import sys
import time
import json
from typing import Dict, Any

from aiohttp import web, ClientSession
from server import PromptServer
from .model_defs import ModelDef

# from lib import loglib
server = PromptServer.instance
app = PromptServer.instance.app
routes = PromptServer.instance.routes

log = logging.getLogger(__name__)


# Set up handlers to pass over to the client and then back.
# This python server extension is simply a middleman from the
# other client which connects to the server and our own webui client
# ------------------------------------------

response = None
response_event = asyncio.Event()


async def handle_uiapi_request(endpoint, request_data=None) -> web.Response:
    log.info(f"/uiapi/{endpoint}")
    response_event.clear()
    
    await server.send_json(f'/uiapi/{endpoint}', request_data or {})

    log.info(f"/uiapi/{endpoint} response_event.wait()")
    await response_event.wait()

    log.info(f"/uiapi/{endpoint} response = {str(response)[:200]}{' ... [TRUNCATED]' if len(str(response)) > 200 else ''}")
    return web.json_response({
        'status': 'ok',
        'response': response
    })


@routes.post('/uiapi/response')
async def uiapi_response(request):
    global response
    log.info("/uiapi/response")
    response = await request.json()
    response_event.set()
    return web.json_response({'status': 'ok'})


@routes.post('/uiapi/get_workflow')
async def uiapi_get_workflow(request):
    return await handle_uiapi_request('get_workflow')

@routes.post('/uiapi/get_fields')
async def uiapi_get_field(request):
    return await handle_uiapi_request('get_fields', await request.json())

@routes.post('/uiapi/set_fields')
async def uiapi_set_field(request):
    return await handle_uiapi_request('set_fields', await request.json())


@routes.post('/uiapi/set_connection')
async def uiapi_set_connection(request):
    return await handle_uiapi_request('set_connection', await request.json())

@routes.post('/uiapi/execute')
async def uiapi_execute(request):
    return await handle_uiapi_request('execute', await request.json())

@routes.post('/uiapi/query_fields')
async def uiapi_query_fields(request):
    return await handle_uiapi_request('query_fields', await request.json())

# Track ongoing downloads
download_tasks: Dict[str, Dict[str, Any]] = {}

async def download_models_task(task_id: str, download_table: Dict[str, dict], workflow: dict) -> None:
    """Background task to handle model downloads"""
    task_info = download_tasks[task_id]
    start_time = time.time()
    
    try:
        nodes = workflow['workflow']['workflow']['nodes']
        
        # First collect all checkpoint names from the workflow
        checkpoints = []
        for node in nodes:
            if 'widgets_values' in node:
                for value in node['widgets_values']:
                    if isinstance(value, str) and any(value.endswith(ext) for ext in ['.safetensors', '.ckpt', '.bin']):
                        checkpoints.append(value)
                        log.info(f"Found checkpoint input {value}")

        total_models = len(checkpoints)
        task_info['total_models'] = total_models
        task_info['current_model'] = 0
        
        # Process downloads one at a time, sequentially
        for idx, ckpt in enumerate(checkpoints, 1):
            name = ckpt.split('/')[-1]
            elapsed = time.time() - start_time
            task_info['current_model'] = idx
            task_info['elapsed_time'] = elapsed
            task_info['progress'][ckpt] = {
                'status': 'downloading',
                'model_number': f"{idx}/{total_models}"
            }
            
            if name in download_table:
                try:
                    model_def = ModelDef(**download_table[name])
                    path = model_def.download(ckpt)
                    task_info['progress'][ckpt].update({
                        'status': 'success',
                        'path': path
                    })
                    log.info(f"Successfully downloaded {ckpt} to {path}")
                    print("")
                except Exception as e:
                    task_info['progress'][ckpt].update({
                        'status': 'error',
                        'error': str(e)
                    })
                    log.error(f"Error downloading {ckpt}: {e}")
            else:
                task_info['progress'][ckpt].update({
                    'status': 'error',
                    'error': 'No download table entry found'
                })
                log.warning(f"No download table entry for {ckpt}")
        
        log.info(f"Model download task completed in {time.time() - start_time:.2f} seconds")
    
    except Exception as e:
        task_info['status'] = 'error'
        task_info['error'] = str(e)
        log.error(f"Download task failed: {e}")
    finally:
        task_info['elapsed_time'] = time.time() - start_time
        task_info['status'] = 'completed'
        task_info['completed'] = True

@routes.post('/uiapi/download_models')
async def uiapi_download_models(request):
    # Get the request data containing the download table
    request_data = await request.json()
    download_table = request_data.get('download_table', {})
    
    # First get the workflow to analyze
    workflow_response = await handle_uiapi_request('get_workflow')
    workflow = workflow_response.body  # Use .text instead of .json()
    workflow = json.loads(workflow)['response']
    
    task_id = str(time.time())
    task_info = {
        'status': 'in_progress',
        'progress': {},
        'completed': False
    }
    download_tasks[task_id] = task_info
    
    # Launch the background task
    asyncio.create_task(download_models_task(task_id, download_table, workflow))
    
    # Clean up old tasks (keep last 10)
    if len(download_tasks) > 10:
        oldest_tasks = sorted(download_tasks.keys())[:-10]
        for old_task in oldest_tasks:
            del download_tasks[old_task]
    
    return web.json_response({
        'status': 'ok',
        'task_id': task_id
    })

@routes.get('/uiapi/download_status/{task_id}')
async def uiapi_download_status(request):
    task_id = request.match_info['task_id']
    if task_id not in download_tasks:
        return web.json_response({
            'status': 'error',
            'error': 'Task not found'
        }, status=404)
    
    return web.json_response(download_tasks[task_id])


