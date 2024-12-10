import asyncio
import logging
import sys
import time
import json
from typing import Dict, Any

from aiohttp import web
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

@routes.post('/uiapi/download_models')
async def uiapi_download_models(request):
    # Get the request data containing the download table
    request_data = await request.json()
    download_table = request_data.get('download_table', {})
    
    # First get the workflow to analyze
    workflow_response = await handle_uiapi_request('get_workflow')
    workflow = json.loads(workflow_response.body)['response']
    nodes = workflow['workflow']['workflow']['nodes']
    
    results = {}
    
    # First collect all checkpoint names from the workflow
    checkpoints = []
    for node in nodes:
        if 'widgets_values' in node:
            widget_values = node['widgets_values']
            for value in widget_values:
                # if value['type'] == 'checkpoint':
                #     checkpoint_inputs.append(value['value'])
                #     log.info(f"Found checkpoint input {value['value']}")

                if isinstance(value, str) and any(value.endswith(ext) for ext in ['.safetensors', '.ckpt', '.bin']):
                    checkpoints.append(value)
                    log.info(f"Found checkpoint input {value}")

    if not checkpoints:
        log.info("No checkpoint inputs found in workflow")
    else:
        log.info(f"Found {len(checkpoints)} checkpoint inputs to process:")
        for input in checkpoints:
            log.info(f"  - {input}")

    # Now process downloads for each checkpoint
    for ckpt in checkpoints:
        print("-"*40)
        name = ckpt.split('/')[-1]
        if name in download_table:
            try:
                model_def = ModelDef(**download_table[name]) # the download table does not specify where exactly to download the model, only matching by name

                path = model_def.download(ckpt)
                results[ckpt] = {
                    'status': 'success',
                    'path': path
                }
            except Exception as e:
                results[ckpt] = {
                    'status': 'error', 
                    'error': str(e)
                }
                log.error(f"Error downloading {ckpt}: {e}")
        else:
            log.warning(f"No download table entry for {ckpt}")
    
    return web.json_response({
        'status': 'ok',
        'downloads': results
    })


