import asyncio
import sys
import time

from aiohttp import web
from server import PromptServer

# from lib import loglib
server = PromptServer.instance
app = PromptServer.instance.app
routes = PromptServer.instance.routes

# log = loglib.make_log('comfyui')
# logerr = loglib.make_logerr('comfyui')


# Set up handlers to pass over to the client and then back.
# This python server extension is simply a middleman from the
# other client which connects to the server and our own webui client
# ------------------------------------------

response = None
response_event = asyncio.Event()


async def handle_uiapi_request(endpoint, request_data=None):
    print(f"/uiapi/{endpoint}")
    response_event.clear()
    
    await server.send_json(f'/uiapi/{endpoint}', request_data or {})

    print(f"/uiapi/{endpoint} response_event.wait()")
    await response_event.wait()

    print(f"/uiapi/{endpoint} response = {response}")
    return web.json_response({
        'status': 'ok',
        'response': response
    })


@routes.post('/uiapi/response')
async def uiapi_response(request):
    global response
    print("/uiapi/response")
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