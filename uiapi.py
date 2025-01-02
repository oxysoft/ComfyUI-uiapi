import asyncio
import logging
import sys
import time
import json
import traceback
from typing import Dict, Any, Optional, Callable, Awaitable, Tuple, List
import base64
from pathlib import Path
import os
import uuid
from dataclasses import dataclass
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install

from aiohttp import web
from aiohttp import web_response
from aiohttp.web import RouteTableDef

import server
from .model_defs import ModelDef
from . import model_defs
from server import PromptServer

routes = PromptServer.instance.routes
server_start_time = time.time()

# Install rich traceback handler
install(show_locals=True)

# Set up rich console
console = Console()

# Configure logging with rich
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True, markup=True, show_time=True, show_path=True)
    ],
)

log = logging.getLogger("uiapi")

# Remove any existing handlers to avoid duplicate logging
for handler in log.handlers[:]:
    log.removeHandler(handler)

# # Add rich handler
# log.addHandler(RichHandler(
#     console=console,
#     rich_tracebacks=True,
#     markup=True,
#     show_time=True,
#     show_path=True
# ))


# Set up handlers to pass over to the client and then back.
# This python server extension is simply a middleman from the
# other client which connects to the server and our own webui client
# ------------------------------------------

INPUT_DIR = Path(__file__).parent.parent.parent / "input"


@dataclass
class PendingRequest:
    """Represents a request that's waiting for client response"""

    request_id: str
    endpoint: str
    data: dict
    event: asyncio.Event
    response: Any = None
    post_process: Optional[Callable[[Any], Awaitable[Any]]] = None


class FrontendManager:
    def __init__(self):
        self._pending_requests: Dict[str, PendingRequest] = {}
        self._webui_ready = asyncio.Event()
        self._buffered_requests: list[PendingRequest] = []
        self._lock = asyncio.Lock()
        log.info("FrontendManager initialized")

    def set_connected(self):
        """Called when ComfyUI web interface connects"""
        log.info(f"(ready) - {len(self._buffered_requests)} requests in buffer")
        self._webui_ready.set()
        # Process buffered requests
        for req in self._buffered_requests:
            log.info(f"Processing buffered request: {req.endpoint}")
            asyncio.create_task(self._send(req))
        self._buffered_requests.clear()

    def set_disconnected(self):
        """Called when ComfyUI web interface disconnects"""
        log.warning("[bold red]WebUI Disconnected[/bold red]")
        self._webui_ready.clear()

    async def _send(self, request: PendingRequest):
        """Send a request to the client"""
        log.info(f"(webui:ask) endpoint={request.endpoint}, id={request.request_id}")
        log.info(f"(webui:ask) {json.dumps(request.data, indent=2)}")
        await server.PromptServer.instance.send_json(request.endpoint, request.data)

    async def send(
        self,
        endpoint: str,
        data: dict = None,
        wait_for_client: bool = False,
        post_process: Optional[Callable[[Any], Awaitable[Any]]] = None,
    ) -> PendingRequest:
        """Create a new request and optionally buffer it"""
        request_id = str(uuid.uuid4())
        endpoint = f"/uiapi/{endpoint}"
        log.info(f"(webui:send) {endpoint} id={request_id}")
        log.info(
            f"(webui:send) wait={wait_for_client}, post-process: {'yes' if post_process else 'no'}"
        )

        if data is None:
            data = {}
        data["request_id"] = request_id

        request = PendingRequest(
            request_id=request_id,
            endpoint=endpoint,
            data=data,
            event=asyncio.Event(),
            post_process=post_process,
        )

        async with self._lock:
            self._pending_requests[request_id] = request
            if wait_for_client and not self._webui_ready.is_set():
                log.info(f"Buffering request {request_id} - WebUI not ready")
                self._buffered_requests.append(request)
                return request

        log.info(f"(webui:send) {request_id}")
        await self._send(request)
        return request

    async def respond(self, request_id: str, response_data: Any):
        """Set response for a request and process it"""
        log.info(f"(webui:respond) {request_id}")
        log.info(f"(webui:respond) {json.dumps(response_data, indent=2)}")

        async with self._lock:
            if request_id not in self._pending_requests:
                log.warning(f"(webui:respond) response received for unknown request: {request_id}")
                return

            request = self._pending_requests[request_id]
            request.response = response_data

            if request.post_process:
                log.info(f"Running post-processing for request {request_id}")
                try:
                    request.response = await request.post_process(response_data)
                    log.info(
                        f"Post-processed response: {json.dumps(request.response, indent=2)}"
                    )
                except Exception as e:
                    log.error(f"Error in post-processing for {request_id}: {e}")
                    log.error(traceback.format_exc())

            request.event.set()
            log.info(f"(webui) request {request_id} completed")

    async def await_response(
        self, request: PendingRequest, timeout: float = 30.0
    ) -> Any:
        """Wait for and return response for a request"""
        log.info(f"(webui:wait) request={request.request_id}, timeout={timeout}s")
        try:
            await asyncio.wait_for(request.event.wait(), timeout)
            log.info(f"(webui:wait) Response received for {request.request_id}")
            return request.response
        except asyncio.TimeoutError:
            log.error(f"(webui:wait) Request {request.request_id} timed out after {timeout}s")
            raise
        finally:
            async with self._lock:
                self._pending_requests.pop(request.request_id, None)
                log.info(f"(webui:wait) Cleaned up request {request.request_id}")


# Initialize the request manager
webui_manager = FrontendManager()


async def handle_uiapi_request(
    endpoint: str,
    request_data: dict = None,
    wait_for_client: bool = False,
    post_process: Optional[Callable[[Any], Awaitable[Any]]] = None,
) -> web.Response:
    """Handle a UI API request with optional post-processing"""
    log.info(f"(/uiapi/{endpoint}")
    log.info(
        f"(/uiapi/{endpoint}) {json.dumps(request_data, indent=2) if request_data else 'None'}"
    )

    try:
        request = await webui_manager.send(
            f"/uiapi/{endpoint}", request_data, wait_for_client, post_process
        )

        if request in webui_manager._buffered_requests:
            log.info(f"Request buffered id={request.request_id}")
            return web.json_response(
                {
                    "status": "pending",
                    "message": "Request buffered until client connects",
                    "request_id": request.request_id,
                }
            )

        response = await webui_manager.await_response(request, timeout=30.0)
        log.info(f"Request completed successfully {request.request_id}")
        return web.json_response({"status": "ok", "response": response})
    except asyncio.TimeoutError:
        log.error(f"Request timeout endpoint={endpoint}")
        return web.json_response(
            {"status": "error", "error": "Request timed out"}, status=408
        )
    except Exception as e:
        log.error(f"Error handling request: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


# Track ongoing downloads
download_tasks: Dict[str, Dict[str, Any]] = {}

async def analyze_workflow_models(workflow: dict) -> Tuple[List[str], List[str], List[str]]:
    """Analyze a workflow and return lists of missing, existing, and all checkpoints.
    
    Args:
        workflow: The workflow to analyze
        
    Returns:
        Tuple of (missing_models, existing_models, all_checkpoints)
    """
    checkpoints = []
    missing_models = []
    existing_models = []
    
    # Extract nodes from workflow
    nodes = workflow.get("nodes", {})
    
    log.info(f"Analyzing workflow with {len(nodes)} nodes")
    
    for node in nodes:
        if isinstance(node, dict) and "widgets_values" in node:
            for value in node["widgets_values"]:
                if isinstance(value, str) and any(
                    value.endswith(ext) for ext in [".safetensors", ".ckpt", ".bin"]
                ):
                    checkpoints.append(value)
                    name = value.split("/")[-1]
                    model_info = model_defs.get_model_info(name)
                    model_type = model_info.get("ckpt_type", None) if model_info else None
                    
                    if model_defs.has_model(model_type, name):
                        path = model_defs.get_model_path(model_type, name)
                        log.info(f"Model {name} already exists at {path}")
                        existing_models.append(name)
                    else:
                        log.info(f"Found missing checkpoint: {value}")
                        missing_models.append(value)
    
    return missing_models, existing_models, checkpoints


@routes.post("/uiapi/webui_ready")
async def uiapi_webui_ready(request):
    """Called when ComfyUI web interface connects"""
    log.info("-> /uiapi/webui_ready - processing any buffered requests")
    webui_manager.set_connected()
    
    try:
        # Fetch current workflow
        request = await webui_manager.send("get_workflow", wait_for_client=True)
        workflow = await webui_manager.await_response(request)
        
        if workflow:
            # Extract nodes from workflow
            nodes = workflow.get("nodes", {})
            checkpoints = []
            missing_models = []
            existing_models = []
            
            log.info(f"Analyzing workflow with {len(nodes)} nodes")
            
            for node in nodes:
                if isinstance(node, dict) and "widgets_values" in node:
                    for value in node["widgets_values"]:
                        if isinstance(value, str) and any(
                            value.endswith(ext) for ext in [".safetensors", ".ckpt", ".bin"]
                        ):
                            checkpoints.append(value)
                            name = value.split("/")[-1]
                            model_info = model_defs.get_model_info(name)
                            model_type = model_info.get("ckpt_type", None) if model_info else None
                            
                            if model_defs.has_model(model_type, name):
                                path = model_defs.get_model_path(model_type, name)
                                log.info(f"Model {name} already exists at {path}")
                                existing_models.append(name)
                            else:
                                log.info(f"Found missing checkpoint: {value}")
                                missing_models.append(value)
            
            log.info("Workflow Analysis Results:")
            log.info(f"Total checkpoints found: {len(checkpoints)}")
            log.info(f"Missing models: {len(missing_models)}")
            for model in missing_models:
                log.info(f"  - {model}")
            log.info(f"Existing models: {len(existing_models)}")
            for model in existing_models:
                log.info(f"  - {model}")
        else:
            log.info("No workflow available to analyze")
            
    except Exception as e:
        log.error(f"Error analyzing workflow: {e}")
        log.error(traceback.format_exc())
    
    return web.json_response({"status": "ok"})


@routes.post("/uiapi/webui_disconnect")
async def uiapi_webui_disconnect(request):
    """Called when ComfyUI web interface disconnects"""
    webui_manager.set_disconnected()
    return web.json_response({"status": "ok"})


@routes.post("/uiapi/response")
async def uiapi_response(request):
    data = await request.json()
    request_id = data.get("request_id")
    log.info("-> /uiapi/response")
    if not request_id:
        return web.json_response(
            {"status": "error", "error": "No request_id provided"}, status=400
        )

    await webui_manager.respond(request_id, data.get("response"))
    return web.json_response({"status": "ok"})


# Example of download_models with post-processing
async def process_workflow_response(response: Any) -> Any:
    """Process workflow response for download_models"""
    if isinstance(response, dict):
        # Handle pending status
        if response.get("status") == "pending":
            return response
        # Handle normal response
        return response.get("response", {})
    elif isinstance(response, web.Response):
        response_text = await response.text()
        response_data = json.loads(response_text)
        if "workflow" in response_data:
            return response_data["workflow"]
    return response


# Add this near other global variables
pending_downloads: Dict[str, Dict] = {}


@routes.post("/uiapi/download_models")
async def uiapi_download_models(request):
    try:
        log.info("-> /uiapi/download_models *")
        # Get the request data containing the download table
        request_data = await request.json()
        download_table = request_data.get("download_table", {})
        workflow = request_data.get("workflow")

        if not workflow:
            log.info("/uiapi/download_models - No workflow provided")
            return web.json_response(
                {"status": "error", "error": "No workflow provided"}, status=400
            )

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Initialize task info
        log.info(f"-> /uiapi/download_models - Initializing task {task_id}")
        download_tasks[task_id] = {
            "status": "initializing",
            "start_time": time.time(),
            "progress": {},
            "completed": False,
        }

        # Start download task in background
        asyncio.create_task(download_models_task(task_id, download_table, workflow))

        log.info(
            f"-> /uiapi/download_models - Task {task_id} started, replying with status..."
        )
        return web.json_response(
            {"status": "ok", "message": "Download task started", "download_id": task_id}
        )

    except Exception as e:
        log.error(f"Download models error: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.get("/uiapi/download_status/{request_id}")
async def uiapi_download_status(request):
    """Handle status requests for downloads in progress"""
    request_id = request.match_info["request_id"]
    log.info(f"-> /uiapi/download_status/{request_id}")

    # First check pending downloads
    if request_id in pending_downloads:
        # Clean up old pending downloads (older than 5 minutes)
        current_time = time.time()
        for rid, info in list(pending_downloads.items()):
            if current_time - info["timestamp"] > 300:  # 5 minutes
                pending_downloads.pop(rid)

        # Return status if still pending
        if request_id in pending_downloads:
            info = pending_downloads[request_id]
            # If task_id is set, redirect to that status
            if info["task_id"]:
                return web.json_response(download_tasks[info["task_id"]])
            return web.json_response(info)

    # Then check active download tasks
    if request_id in download_tasks:
        return web.json_response(download_tasks[request_id])

    return web.json_response({"status": "error", "error": "Task not found"}, status=404)


@routes.post("/uiapi/get_workflow")
async def uiapi_get_workflow(request):
    log.info("-> /uiapi/get_workflow")
    return await handle_uiapi_request("get_workflow")


@routes.post("/uiapi/get_fields")
async def uiapi_get_field(request):
    log.info("-> /uiapi/get_fields")
    return await handle_uiapi_request("get_fields", await request.json())


def save_base64_image(base64_data: str, filepath: Path) -> None:
    """Save base64 image data to a file"""
    try:
        image_data = base64.b64decode(base64_data)
        os.makedirs(filepath.parent, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(image_data)
    except Exception as e:
        log.error(f"Failed to save base64 image: {e}")
        import traceback

        log.error(traceback.format_exc())
        raise


@routes.post("/uiapi/set_fields")
async def uiapi_set_field(request):
    request_data = await request.json()
    log.info("-> /uiapi/set_fields")

    # Process any image fields before passing to the main handler
    if "fields" in request_data:
        for field in request_data["fields"]:
            if isinstance(field[1], dict) and field[1].get("type") == "image_base64":
                path = field[0]
                # Generate input name from the node path
                input_name = f'INPUT_{path.split(".")[0]}.png'

                # TODO: Verify this is the correct path for ComfyUI's input folder
                filepath = INPUT_DIR / input_name

                # Save the base64 image
                save_base64_image(field[1]["data"], filepath)

                # Replace the base64 data with just the filename
                field[1] = input_name

    return await handle_uiapi_request("set_fields", request_data)


@routes.post("/uiapi/set_connection")
async def uiapi_set_connection(request):
    log.info("-> /uiapi/set_connection")
    return await handle_uiapi_request("set_connection", await request.json())


@routes.post("/uiapi/execute")
async def uiapi_execute(request):
    log.info("-> /uiapi/execute")
    return await handle_uiapi_request("execute", await request.json())


@routes.post("/uiapi/query_fields")
async def uiapi_query_fields(request):
    log.info("-> /uiapi/query_fields")
    return await handle_uiapi_request("query_fields", await request.json())


@routes.post("/uiapi/add_model_url")
async def uiapi_add_model_url(request):
    """Add a URL for a model that wasn't in the download table"""
    log.info("-> /uiapi/add_model_url")
    try:
        data = await request.json()
        model_name = data.get("model_name")
        model_url = data.get("url")
        model_type = data.get("model_type", "model")  # Default to 'model' type

        if not model_name or not model_url:
            return web.json_response(
                {"status": "error", "error": "Missing model_name or url"}, status=400
            )

        # Create model definition
        model_def = {"url": model_url, "ckpt_type": model_type}

        return web.json_response({"status": "ok", "model_def": model_def})

    except Exception as e:
        log.error(f"Error adding model URL: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


async def download_models_task(task_id: str, download_table: Dict[str, dict], workflow: dict) -> None:
    """Background task to handle model downloads"""
    log.info(f"Starting download task {task_id}")

    task_info = download_tasks[task_id]
    start_time = time.time()

    try:
        # Extract nodes from workflow
        nodes = workflow["nodes"]

        log.info(
            f"/uiapi/download_models - Task {task_id} - Extracted {len(nodes)} nodes from workflow"
        )

        # First collect all checkpoint names from the workflow
        checkpoints = []
        missing_models = []
        existing_models = []
        
        task_info.update({
            "status": "checking",
            "progress": {},
        })
        
        for node in nodes:
            if isinstance(node, dict) and "widgets_values" in node:
                for value in node["widgets_values"]:
                    if isinstance(value, str) and any(
                        value.endswith(ext) for ext in [".safetensors", ".ckpt", ".bin"]
                    ):
                        checkpoints.append(value)
                        name = value.split("/")[-1]
                        model_info = download_table.get(name, {})
                        model_type = model_info.get("ckpt_type", None)
                        
                        if model_defs.has_model(model_type, name):
                            path = model_defs.get_model_path(None, name)
                            log.info(f"Model {name} already exists at {path}, skipping")
                            existing_models.append(name)
                            task_info["progress"][value] = {
                                "status": "success",
                                "path": str(path),
                                "model_number": "existing"
                            }
                        else:
                            missing_models.append(value)
                            log.info(f"Found missing checkpoint input {value}")

        total_models = len(missing_models)
        log.info(f"Found {total_models} models to download")

        if total_models == 0:
            log.info("All models already exist, nothing to download")
            task_info.update({
                "status": "completed",
                "completed": True,
                "elapsed_time": time.time() - start_time
            })
            return

        task_info.update({
            "total_models": total_models,
            "current_model": 0,
            "status": "downloading",
        })

        # Get URLs only for missing models
        log.info(f"Requesting URLs for {len(missing_models)} missing models")
        request = await webui_manager.send(
            "get_model_url",
            {
                "model_names": missing_models,
                "existing_models": [ckpt.split("/")[-1] for ckpt in checkpoints if ckpt not in missing_models]
            },
            wait_for_client=True
        )

        # Wait for response from webui
        url_map = await webui_manager.await_response(request)
        if not url_map:
            log.warning("No URLs provided for models")
            task_info.update({
                "status": "error",
                "error": "No URLs provided for models"
            })
            return

        # Process downloads one at a time
        for idx, ckpt in enumerate(missing_models, 1):
            name = ckpt.split("/")[-1]
            elapsed = time.time() - start_time

            log.info(f"Processing model {idx}/{len(missing_models)}: {name}")

            task_info["current_model"] = idx
            task_info["elapsed_time"] = elapsed
            task_info["progress"][ckpt] = {
                "status": "downloading",
                "model_number": f"{idx}/{len(missing_models)}",
            }

            try:
                if name not in download_table and name not in url_map:
                    log.warning(f"No URL provided for {name}, skipping...")
                    task_info["progress"][ckpt].update(
                        {"status": "error", "error": "No download URL provided"}
                    )
                    continue

                # Use URL from batch response if available, otherwise from download_table
                model_info = download_table.get(name, {})
                if name in url_map:
                    model_info["url"] = url_map[name]

                log.info(f"Creating ModelDef for {name}")
                model_def = ModelDef(**model_info)

                log.info(f"Starting download of {name}")
                path = model_def.download(ckpt)
                task_info["progress"][ckpt].update(
                    {"status": "success", "path": str(path)}
                )
                log.info(f"Successfully downloaded {ckpt} to {path}")

            except Exception as e:
                log.error(f"Error downloading {ckpt}: {e}")
                log.error(traceback.format_exc())
                task_info["progress"][ckpt].update({"status": "error", "error": str(e)})

        task_info["status"] = "completed"
        task_info["completed"] = True
        task_info["elapsed_time"] = time.time() - start_time
        log.info(
            f"Download task {task_id} completed in {task_info['elapsed_time']:.1f}s"
        )

    except Exception as e:
        log.error(f"Download task {task_id} failed: {e}")
        log.error(traceback.format_exc())
        task_info["status"] = "error"
        task_info["error"] = str(e)


@routes.get("/uiapi/connection_status")
async def uiapi_connection_status(request):
    """Check if ComfyUI web interface is connected and get system status"""
    try:
        status = {
            "status": "ok",
            "webui_connected": webui_manager._webui_ready.is_set(),
            "pending_requests": len(webui_manager._buffered_requests),
            "active_downloads": len(download_tasks),
            "pending_downloads": len(pending_downloads),
            "server_time": time.time(),
            "server_uptime": time.time() - server_start_time,
        }

        log.info("-> /uiapi/connection_status")
        # log.info("-> /uiapi/connection_status: {status}")

        return web.json_response(status)
    except Exception as e:
        log.error(f"Error in connection status check: {e}")
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.post("/uiapi/client_disconnect")
async def uiapi_client_disconnect(request):
    """Handle explicit client disconnect notification"""
    try:
        webui_manager.set_disconnected()
        log.info("Client explicitly disconnected")
        return web.json_response({"status": "ok"})
    except Exception as e:
        log.error(f"Error handling client disconnect: {e}")
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.post("/uiapi/download_model")
async def uiapi_download_model(request):
    """Handle single model download request"""
    try:
        data = await request.json()
        model_name = data.get("model_name")
        model_def = data.get("model_def")

        if not model_name or not model_def:
            return web.json_response(
                {"status": "error", "error": "Missing model_name or model_def"},
                status=400,
            )

        # Create ModelDef instance
        try:
            model = ModelDef(**model_def)
        except Exception as e:
            return web.json_response(
                {"status": "error", "error": f"Invalid model definition: {str(e)}"},
                status=400,
            )

        # Start download
        try:
            path = model.download(model_name)
            return web.json_response({"status": "success", "path": str(path)})
        except Exception as e:
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    except Exception as e:
        log.error(f"Download model error: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.get("/uiapi/model_status/{model_name}")
async def uiapi_model_status(request):
    """Get status of a specific model"""
    model_name = request.match_info["model_name"]

    try:
        # Check if model exists
        for model_type in model_defs.MODEL_PATHS.keys():
            if model_defs.has_model(model_type, model_name):
                path = model_defs.get_model_path(model_type, model_name)
                return web.json_response(
                    {
                        "status": "success",
                        "exists": True,
                        "path": str(path),
                        "type": model_type,
                    }
                )

        # Check if download is in progress
        if model_name in download_tasks:
            task = download_tasks[model_name]
            return web.json_response(
                {
                    "status": "downloading",
                    "progress": task.get("progress", 0),
                    "started": task.get("start_time"),
                    "elapsed": time.time() - task.get("start_time", time.time()),
                }
            )

        return web.json_response({"status": "not_found", "exists": False})

    except Exception as e:
        log.error(f"Model status error: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.post("/uiapi/get_model_url")
async def uiapi_get_model_url(request):
    """Handle requests to get a model's download URL"""
    try:
        data = await request.json()
        model_name = data.get("model_name")
        model_names = data.get("model_names")
        log.info("-> /uiapi/get_model_url")
        log.info(f"-> /uiapi/get_model_url: {data}")

        # Handle batch request
        if model_names:
            if not isinstance(model_names, list):
                return web.json_response(
                    {"status": "error", "error": "model_names must be a list"}, status=400
                )
            request = await webui_manager.send(
                "get_model_url",
                {"model_names": model_names},
                wait_for_client=True
            )
            response = await webui_manager.await_response(request)
            return web.json_response({"status": "ok", "urls": response})

        # Handle single model request
        if not model_name:
            return web.json_response(
                {"status": "error", "error": "Missing model_name"}, status=400
            )

        # Create a request to get URL from webui
        request = await webui_manager.send(
            "get_model_url",
            {"model_name": model_name},
            wait_for_client=True
        )

        # Wait for response from webui
        response = await webui_manager.await_response(request)

        if not response or not response.get("url"):
            return web.json_response(
                {"status": "error", "error": "No URL provided"}, status=400
            )

        return web.json_response(
            {
                "status": "ok",
                "url": response["url"],
                "model_type": response.get("model_type", "model"),
            }
        )

    except Exception as e:
        log.error(f"Error getting model URL: {e}")
        log.error(traceback.format_exc())
        return web.json_response({"status": "error", "error": str(e)}, status=500)
