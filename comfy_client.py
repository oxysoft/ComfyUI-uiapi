import asyncio
import base64
import io
import json
import logging
import math
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable, Dict, Optional, cast, Union, TypedDict, Mapping, BinaryIO

import cv2
import numpy as np
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from websockets import WebSocketClientProtocol
from websockets.sync.client import connect
from websockets.sync.client import ClientConnection

from model_defs import ModelDef


# Type definitions for better type hints
# ----------------------------------------
class ImageInfo(TypedDict):
    """Information about an image in the workflow"""
    filename: str
    subfolder: str
    type: str

class UploadResponse(TypedDict):
    """Response from uploading an image"""
    name: str
    subfolder: str
    type: str

class WorkflowResponse(TypedDict):
    """Response from queueing a workflow"""
    prompt_id: str

class HistoryOutput(TypedDict):
    """Output information from workflow history"""
    images: list[ImageInfo]

class HistoryData(TypedDict):
    """History data for a workflow execution"""
    outputs: Dict[str, HistoryOutput]

class ExecutionResult(TypedDict):
    """Complete result from executing a workflow"""
    prompt_id: str
    outputs: Dict[str, list[np.ndarray]]
    history: HistoryData

# Set up rich console
console = Console()

# # Configure logging with rich
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(message)s",
#     datefmt="[%X]",
#     handlers=[RichHandler(
#         rich_tracebacks=True,
#         markup=True,
#         show_time=True,
#         show_path=True
#     )]
# )

log = logging.getLogger("comfy_client")
PNG_MAGIC_BYTES = b"\x89PNG\r\n\x1a\n"


# Allows using uiapi plugin to directly control the comfy webui
# ------------------------------------------------------------
def is_image(value: Any) -> bool:
    """Check if a value is an image type we can handle"""
    is_image = isinstance(value, np.ndarray) or isinstance(value, Image.Image)
    if is_image:
        return True

    import torch
    if isinstance(value, torch.Tensor):
        return True

    return False


def clamp01(v):
    return min(max(v, 0), 1)


def encode_image_to_base64(image):
    """Convert various image types to base64 string"""
    if isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    elif isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        if image.dtype in [np.float32, np.float64]:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer.tobytes()).decode()
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


class ComfyClient:
    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws: Optional[ClientConnection] = None
        self._webui_ready = False
        self._connection_lock = asyncio.Lock()
        self._reconnect_backoff = 1.0  # Initial backoff in seconds
        self._sync_lock = threading.Lock()  # Add thread safety for sync operations
        log.info(f"""[bold green]ComfyClient initialized[/bold green]
            Server: {server_address}
            Client ID: {self.client_id}""")

    def ensure_connection(self):
        """Ensure both HTTP and WebSocket connections are alive"""
        return self.run_sync(self.ensure_connection_async())

    async def ensure_connection_async(self, require_uiapi=True) -> bool:
        """
        Ensure both HTTP and WebSocket connections are alive.
        After this call completes successfully, self.ws is guaranteed to be a valid WebSocket.
        """
        async with self._connection_lock:
            while True:
                try:
                    if not self._webui_ready:
                        log.info("Establishing connection with ComfyUI...")

                    if require_uiapi:
                        # Check/establish HTTP connection
                        response = await self._make_request_once(
                            "GET", "/uiapi/connection_status"
                        )
                        if not response.get("webui_connected"):
                            raise ConnectionError("WebUI not connected")

                    # Check/establish WebSocket connection
                    if not self.ws:
                        log.info(
                            "[yellow]Establishing WebSocket connection...[/yellow]"
                        )
                        ws_url = (
                            f"ws://{self.server_address}/ws?clientId={self.client_id}"
                        )
                        self.ws = connect(ws_url, open_timeout=10)

                    self._webui_ready = True
                    assert self.ws is not None
                    return True

                except Exception as e:
                    self._webui_ready = False
                    wait_time = min(self._reconnect_backoff * 2, 30)
                    log.warning(f"""[yellow]Connection attempt failed:[/yellow]
                        Error: {str(e)}
                        Retrying in {wait_time:.1f}s...""")
                    await asyncio.sleep(wait_time)
                    self._reconnect_backoff = wait_time

    async def _make_request_once(
        self, 
        method: str, 
        url: str, 
        data: Optional[dict] = None, 
        files: Optional[Mapping[str, Union[BinaryIO, tuple[Optional[str], BinaryIO, str]]]] = None,
        verbose: bool = False
    ) -> Union[Dict[str, Any], np.ndarray]:
        """Single attempt at making a request without retry logic"""
        address = f"{self.server_address}{url}"
        if not address.startswith("http"):
            address = f"http://{address}"

        if verbose:
            log.debug(
                f"[cyan]Request details:[/cyan]\n"
                + f"  URL: {address}\n"
                + f"  Method: {method}\n"
                + f"  Data: {data if data else 'None'}\n"
                + f"  Files: {files.keys() if files else 'None'}"
            )

        if files:
            # Handle multipart/form-data for file uploads
            import requests
            response = requests.post(address, data=data, files=files)
            response.raise_for_status()
            ret = response.json()
            return cast(Dict[str, Any], ret)
        else:
            # Handle regular JSON requests
            headers = {"Content-Type": "application/json"}
            data_bytes = None
            if data is not None:
                data_bytes = json.dumps(data).encode("utf-8")

            req = urllib.request.Request(
                address, headers=headers, data=data_bytes, method=method
            )
            with urllib.request.urlopen(req) as response:
                ret = response.read()
                if isinstance(ret, str):
                    return cast(Dict[str, Any], json.loads(ret))
                elif isinstance(ret, bytes):
                    # check if the response is png
                    if ret.startswith(PNG_MAGIC_BYTES):
                        img = cv2.imdecode(np.frombuffer(ret, np.uint8), cv2.IMREAD_COLOR)
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        return cast(Dict[str, Any], json.loads(ret.decode("utf-8")))
                else:
                    raise ValueError(f"Unexpected response type: {type(ret)}")

    async def _make_request(
        self, 
        method: str, 
        url: str, 
        data: Optional[dict] = None,
        files: Optional[Mapping[str, Union[BinaryIO, tuple[Optional[str], BinaryIO, str]]]] = None,
        verbose: bool = False
    ) -> Union[Dict[str, Any], np.ndarray]:
        """Make request with automatic reconnection"""
        while True:
            try:
                await self.ensure_connection_async(require_uiapi='uiapi' in url)
                return await self._make_request_once(method, url, data, files, verbose)
            except Exception as e:
                log.error(f"""[red]Request failed:[/red]
                    Method: {method}
                    URL: {url}
                    Error: {str(e)}
                    Retrying...""")
                traceback.print_exc()
                self._webui_ready = False
                continue

    @classmethod
    def ConnectNew(
        cls, server_address: str, timeout: Optional[float] = None
    ) -> "ComfyClient":
        """Synchronous connection method"""
        # Create event loop if there isn't one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(cls.ConnectNewAsync(server_address, timeout))

    @classmethod
    async def ConnectNewAsync(
        cls, server_address: str, timeout: Optional[float] = None
    ) -> "ComfyClient":
        """Async connection method"""
        client = cls(server_address)
        try:
            start_time = time.time()
            while True:
                try:
                    await client.ensure_connection_async()
                except Exception as e:
                    if timeout and (time.time() - start_time) > timeout:
                        raise TimeoutError(f"Connection timeout after {timeout}s")
                    continue
        except Exception as e:
            log.error(f"[bold red]Failed to establish connection:[/bold red] {str(e)}")
            raise

        return client

    def _make_sync_wrapper(self, async_func) -> Callable:
        """Create a sync version of an async function"""

        def sync_wrapper(*args, **kwargs):
            coroutine = async_func(*args, **kwargs)
            self.run_sync(coroutine)

        return sync_wrapper

    def run_sync(self, coroutine):
        """Run a coroutine synchronously"""
        with self._sync_lock:  # Thread safety for sync operations
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(coroutine)

    async def await_execution(self, require_uiapi=False):
        """Wait for execution with automatic reconnection"""
        start_time = time.time()
        max_timeout = 90
        last_status_time = 0
        got_execution_status = False

        while True:
            try:
                await self.ensure_connection_async(require_uiapi=require_uiapi)
                assert self.ws is not None

                out = self.ws.recv()
                current_time = time.time()
                elapsed = current_time - start_time

                if current_time - last_status_time >= 5:
                    log.info(
                        f"[cyan]Execution status:[/cyan] Running for {elapsed:.1f}s"
                    )
                    last_status_time = current_time


                # Handle messages
                # ----------------------------------------
                if isinstance(out, str):
                    msg = json.loads(out)
                    msg_type = msg["type"]
                    
                    if msg_type == "status":
                        queue_size = msg["data"]["status"]["exec_info"]["queue_remaining"]
                        if queue_size > 0:
                            got_execution_status = True
                        elif got_execution_status:
                            log.info(f"[green]Execution completed[/green] in {elapsed:.1f}s")
                            pass
                            
                    elif msg_type == "execution_start":
                        prompt_id = msg["data"]["prompt_id"]
                        log.info(f"[cyan](prompt: {prompt_id}) [/cyan]")
                        
                    elif msg_type == "execution_cached":
                        cached_nodes = len(msg["data"]["nodes"])
                        log.info(f"[cyan](cached: {cached_nodes}) [/cyan]")
                        
                    elif msg_type == "executed":
                        node = msg["data"]["node"]
                        if "output" in msg["data"] and msg["data"]["output"]:
                            if "images" in msg["data"]["output"]:
                                images = msg["data"]["output"]["images"]
                                log.info(f"[cyan]({node})[/cyan] generated {len(images)} images")
                            
                    elif msg_type == "executing":
                        node = msg["data"]["node"]
                        if node:
                            log.info(f"[cyan]({node})[/cyan] executing")
                    elif msg_type == "progress":
                        value = msg["data"]["value"]
                        max_val = msg["data"]["max"]
                        node = msg["data"]["node"]
                        log.info(f"[cyan]({node})[/cyan] step {value}/{max_val}")
                            
                    elif msg_type == "execution_success":
                        # log.info(f"[green]Execution completed[/green] in {elapsed:.1f}s")
                        pass
                        
                    elif msg_type.startswith("crystools"):
                        continue
                else:
                    pass # we can also get some image data here of the preview, which could be interesting

            except Exception as e:
                log.warning(f"""[yellow]Execution monitoring interrupted:[/yellow]
                    Error: {str(e)}
                    Elapsed: {elapsed:.1f}s
                    Attempting to restore connection...""")
                self._webui_ready = False
                traceback.print_exc()
                continue

    def free(self):
        self.freed = True

    async def get_workflow(self):
        """Get current workflow state"""
        return await self.json_post_async("/uiapi/get_workflow")

    def is_valid_value(self, value: Any) -> bool:
        """
        Check if a value is valid to send to ComfyUI.
        Prevents sending NaN, Infinity, and other problematic values.
        """
        if isinstance(value, (int, float)):
            return not (math.isinf(value) or math.isnan(value))
        elif isinstance(value, (str, bool)):
            return True
        elif value is None:
            return False
        elif isinstance(value, (list, tuple)):
            return all(self.is_valid_value(v) for v in value)
        return True

    # SYNC WRAPPERS
    # ----------------------------------------

    def json_post(
        self, url: str, input_data: Optional[dict] = None, verbose: bool = False
    ) -> dict:
        """Sync version of json_post_async"""
        return self.run_sync(self.json_post_async(url, input_data, verbose))

    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Sync version of model status check"""
        return self.run_sync(self.get_model_status_async(model_name))

    def download_models(self, models: Dict[str, ModelDef], timeout: int = 300):
        """Sync version that calls async version"""
        return self.run_sync(self.download_models_async(models, timeout))

    def download_model(
        self, model_name: str, model_def: ModelDef, timeout: int = 300
    ) -> Dict[str, Any]:
        """Sync version of direct model download"""
        return self.run_sync(self.download_model_async(model_name, model_def, timeout))

    def gets(self, verbose=False):
        """Sync version of gets_async"""
        return self.run_sync(self.gets_async(verbose))

    def get(self, path_or_paths, verbose=False):
        """Sync version of get_field_async"""
        return self.run_sync(self.get_async(path_or_paths, verbose))

    def set(self, path_or_fields, value=None, verbose=False, clamp=False):
        """Sync version of set_async"""
        return self.run_sync(self.set_async(path_or_fields, value, verbose, clamp))

    def connect(self, path1, path2, verbose=False):
        """Sync version of connect_async"""
        return self.run_sync(self.connect_async(path1, path2, verbose))

    def execute(self, wait=True):
        """Sync version of execute_async"""
        return self.run_sync(self.execute_async(wait))

    # ASYNC API
    # ----------------------------------------

    async def json_get_async(self, url: str, verbose: bool = False) -> Any:
        """Make GET request with retry logic"""
        return await self._make_request("GET", url, None, verbose)

    async def json_post_async(
        self, url: str, input_data: Optional[dict] = None, verbose: bool = False
    ) -> Any:
        """Make POST request with retry logic"""
        data = input_data or {}
        if isinstance(data, str):
            data = json.loads(data)

        data["verbose"] = verbose
        data["client_id"] = self.client_id

        return await self._make_request("POST", url, data, verbose)

    async def download_models_async(
        self, models: Dict[str, ModelDef], timeout: int = 300
    ) -> Dict[str, Dict[str, Any]]:
        """
        Download models with automatic retry and progress tracking.
        """
        download_table = {
            filename: model_def.to_dict() for filename, model_def in models.items()
        }

        log.info(f"""[cyan]Starting model downloads:[/cyan]
            Models to download: {len(models)}
            Timeout per model: {timeout}s""")

        try:
            # Start the download process
            response = await self._make_request(
                "POST", "/uiapi/download_models", {"download_table": download_table}
            )

            if not isinstance(response, dict):
                raise ValueError(f"Unexpected response format: {response}")

            # Get download ID from response
            download_id = response.get("download_id")
            if not download_id:
                raise ValueError("No download_id in response")

            # Track download progress
            start_time = time.time()
            last_status_time = 0

            while True:
                try:
                    # Use the download status endpoint
                    status = await self._make_request(
                        "GET", f"/uiapi/download_status/{download_id}"
                    )
                    current_time = time.time()
                    elapsed = current_time - start_time

                    # Log status every 5 seconds
                    if current_time - last_status_time >= 5:
                        progress = status.get("progress", {})
                        completed = sum(
                            1
                            for info in progress.values()
                            if info.get("status") == "success"
                        )
                        total = len(models)
                        log.info(f"""[cyan]Download status:[/cyan]
                            Progress: {completed}/{total} models
                            Elapsed time: {elapsed:.1f}s""")
                        last_status_time = current_time

                    # Check if downloads are complete
                    if status.get("completed", False):
                        log.info(
                            f"[green]All downloads completed[/green] in {elapsed:.1f}s"
                        )
                        return status.get("progress", {})

                    # Check for any errors
                    for model, info in status.get("progress", {}).items():
                        if info.get("status") == "error":
                            log.error(
                                f"[red]Error downloading {model}:[/red] {info.get('error')}"
                            )

                    await asyncio.sleep(1)

                    # Check timeout
                    if elapsed > timeout:
                        raise TimeoutError(f"Download timeout after {timeout}s")

                except Exception as e:
                    if isinstance(e, TimeoutError):
                        raise
                    log.warning(f"""[yellow]Download status check failed:[/yellow]
                        Error: {str(e)}
                        Elapsed: {elapsed:.1f}s
                        Attempting to continue...""")
                    continue

        except Exception as e:
            log.error(f"""[red]Download process failed:[/red]
                Error: {str(e)}
                Total time: {time.time() - start_time:.1f}s""")
            raise

    async def download_model_async(
        self, model_name: str, model_def: ModelDef, timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Download a single model directly without needing workflow context.

        Args:
            model_name: Name of the model file
            model_def: ModelDef object containing download info
            timeout: Maximum time to wait for download
        """
        try:
            log.info(f"""[cyan]Starting direct model download:[/cyan]
                Model: {model_name}
                Type: {model_def.ckpt_type}
                Timeout: {timeout}s""")

            response = await self._make_request(
                "POST",
                "/uiapi/download_model",
                {"model_name": model_name, "model_def": model_def.to_dict()},
            )

            if not isinstance(response, dict):
                raise ValueError(f"Unexpected response format: {response}")

            # Monitor download progress
            start_time = time.time()
            last_status_time = 0

            while True:
                try:
                    status = await self._make_request(
                        "GET", f"/uiapi/model_status/{model_name}"
                    )
                    current_time = time.time()
                    elapsed = current_time - start_time

                    # Log status every 5 seconds
                    if current_time - last_status_time >= 5:
                        log.info(f"""[cyan]Download status for {model_name}:[/cyan]
                            Status: {status.get('status', 'unknown')}
                            Elapsed time: {elapsed:.1f}s""")
                        last_status_time = current_time

                    if status.get("status") == "success":
                        log.info(f"[green]Download completed[/green] in {elapsed:.1f}s")
                        return status

                    if status.get("status") == "error":
                        error = status.get("error", "Unknown error")
                        log.error(f"[red]Download failed:[/red] {error}")
                        raise ValueError(error)

                    await asyncio.sleep(1)

                    if elapsed > timeout:
                        raise TimeoutError(f"Download timeout after {timeout}s")

                except Exception as e:
                    if isinstance(e, (TimeoutError, ValueError)):
                        raise
                    log.warning(f"""[yellow]Status check failed:[/yellow]
                        Error: {str(e)}
                        Elapsed: {elapsed:.1f}s
                        Attempting to continue...""")
                    continue

        except Exception as e:
            log.error(f"""[red]Direct download failed for {model_name}:[/red]
                Error: {str(e)}""")
            raise

    async def get_model_status_async(self, model_name: str) -> Dict[str, Any]:
        """
        Get status of a specific model.

        Args:
            model_name: Name of the model file to check

        Returns:
            Dict containing model status information
        """
        try:
            return await self._make_request("GET", f"/uiapi/model_status/{model_name}")
        except Exception as e:
            log.error(
                f"[red]Failed to get model status for {model_name}:[/red] {str(e)}"
            )
            raise

    async def gets_async(self, verbose=False):
        """
        Query all available fields from the ComfyUI workflow.

        This function sends a request to the ComfyUI server to retrieve information
        about all available fields in the current workflow. It uses the '/uiapi/query_fields'
        endpoint to fetch this data.

        Returns:
            dict: A dictionary containing information about all available fields in the workflow.
                  The structure of this dictionary depends on the ComfyUI server's response.

        Raises:
            ValueError: If the server response is not in the expected format.
        """
        log.info("query_fields")
        log.info(f"client_id: {self.client_id}")

        response = self.json_post_async("/uiapi/query_fields", verbose=verbose)

        log.info(f"query_fields response: {response}")
        log.info(response)

        if isinstance(response, dict):
            if "result" in response:
                return response["result"]
            elif "response" in response:
                return response["response"]
            else:
                raise ValueError("Unexpected response format from server")
        else:
            raise ValueError("Unexpected response format from server")

    async def get_async(self, path_or_paths, verbose=False):
        """Get field values with proper async handling"""
        is_single = isinstance(path_or_paths, str)
        paths = [path_or_paths] if is_single else path_or_paths

        try:
            response = await self.json_post_async(
                "/uiapi/get_fields",
                {
                    "fields": paths,
                },
                verbose=verbose,
            )

            if isinstance(response, dict):
                if "result" in response:
                    result = response["result"]
                    return result[path_or_paths] if is_single else result
                elif "response" in response:
                    result = response["response"]
                    return result[path_or_paths] if is_single else result

            raise ValueError(f"Unexpected response format: {response}")
        except Exception as e:
            log.error(f"[red]Failed to get field values:[/red] {str(e)}")
            raise

    async def set_async(self, path_or_fields, value=None, verbose=False, clamp=False):
        """Set field values with proper async handling"""
        try:
            fields = (
                [(path_or_fields, value)]
                if isinstance(path_or_fields, str)
                else path_or_fields
            )
            processed_fields = []
            processed_paths = []

            for path, val in fields:
                if not self.is_valid_value(val):
                    log.warning(f"Skipping invalid value for {path}: {val}")
                    continue

                if is_image(val):
                    try:
                        base64_img = encode_image_to_base64(val)
                        processed_paths.append(path)
                        processed_fields.append(
                            [path, {"type": "image_base64", "data": base64_img}]
                        )
                    except Exception as e:
                        log.error(f"Failed to encode image for {path}: {e}")
                        continue
                else:
                    if clamp:
                        val = clamp01(val)
                    processed_paths.append(path)
                    processed_fields.append([path, val])

            if not processed_fields:
                log.warning("No valid fields to set")
                return None

            log.info(f"Setting fields: {processed_paths}")
            return await self.json_post_async(
                "/uiapi/set_fields",
                {
                    "fields": processed_fields,
                },
                verbose=verbose,
            )
        except Exception as e:
            log.error(f"[red]Failed to set fields:[/red] {str(e)}")
            raise

    async def connect_async(self, path1, path2, verbose=False):
        return await self.json_post_async(
            "/uiapi/set_connection",
            {
                "field": [path1, path2],
            },
            verbose=verbose,
        )

    async def execute_async(self, wait=True):
        """
        Execute the workflow and return the image of the final SaveImage node
        """
        try:
            await self.ensure_connection_async()

            log.info("[cyan]Executing prompt...[/cyan]")
            ret = await self.json_post_async("/uiapi/execute")

            if not wait:
                return ret

            if not isinstance(ret, dict):
                raise ValueError(f"Unexpected response format: {ret}")

            # Handle broken responses with retry
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                if "response" not in ret or "prompt_id" not in ret.get("response", {}):
                    retry_count += 1
                    log.warning(
                        f"[yellow]Broken response (attempt {retry_count}/{max_retries}), retrying...[/yellow]"
                    )
                    log.debug(f"Response was: {ret}")
                    await asyncio.sleep(1)  # Brief delay before retry
                    await (
                        self.ensure_connection_async()
                    )  # This handles both HTTP and WS connections
                    ret = await self.json_post_async("/uiapi/execute")
                else:
                    break

            if retry_count == max_retries:
                raise ValueError("Maximum retries exceeded for broken responses")

            exec_id = ret["response"]["prompt_id"]

            # minimum execution time to prevent race condition
            # where we get a signal that the queue is empty before the execution has posted
            # this could be fixed by correctly tracking render IDs, but it doesn't seem like comfyui has it.
            await asyncio.sleep(3)

            await self.await_execution()

            log.info("[green]Execution completed, fetching results...[/green]")

            workflow_json = await self.get_workflow()
            if not isinstance(workflow_json, dict) or "response" not in workflow_json:
                raise ValueError(f"Invalid workflow response: {workflow_json}")

            output_node_id = self.find_output_node(workflow_json["response"])
            if not output_node_id:
                raise ValueError("No output node found in workflow")

            history = await self.get_history_async(exec_id)
            history_data = history[exec_id]

            filenames = history_data["outputs"][output_node_id]["images"]
            if not filenames:
                log.warning("[yellow]No images found in execution output[/yellow]")
                return None

            info = filenames[0]
            return await self.get_image_async(
                info["filename"], info["subfolder"], info["type"]
            )

        except Exception as e:
            log.error(f"[red]Execution failed:[/red] {str(e)}")
            raise



    async def get_history_async(self, prompt_id):
        """Get execution history with retry logic"""
        return await self._make_request("GET", f"/history/{prompt_id}")

    async def get_image_async(self, filename, subfolder, folder_type):
        """Get image with retry logic"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)

        try:
            ret = await self._make_request("GET", f"/view?{url_values}")
            return ret
        except Exception as e:
            log.error(f"[red]Failed to get image:[/red] {str(e)}")
            raise

    def _find_node_by_title_or_id(self, workflow: dict, node_ref: str) -> Optional[tuple[str, dict]]:
        """Find a node by its title or ID in the workflow
        
        Args:
            workflow: The workflow dict
            node_ref: Either a node ID or title to search for
            
        Returns:
            Tuple of (node_id, node_dict) if found, None otherwise
        """
        # First try direct node ID lookup
        if node_ref in workflow:
            return node_ref, workflow[node_ref]
            
        # Then try matching by title
        for node_id, node in workflow.items():
            if node.get('_meta', {}).get('title', '').lower() == node_ref.lower():
                return node_id, node
        return None

    def _get_default_input_name(self, node: dict) -> Optional[str]:
        """Get the name of the first input field for a node"""
        if not node.get('inputs'):
            return None
        return next(iter(node['inputs'].keys()), None)

    async def execute_workflow_async(
        self, 
        workflow: dict, 
        fields: Optional[Dict[str, Any]] = None, 
        wait: bool = True
    ) -> Union[WorkflowResponse, ExecutionResult]:
        """Execute a workflow directly using the JSON API format."""
        try:
            # Convert any None values to empty strings
            def convert_nones(obj):
                if isinstance(obj, dict):
                    return {k: convert_nones(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_nones(x) for x in obj]
                elif obj is None:
                    return ""
                return obj

            workflow = cast(dict, convert_nones(workflow))

            await self.ensure_connection_async(require_uiapi=False)
            
            # Deep copy workflow to avoid modifying original
            workflow = json.loads(json.dumps(workflow))
            
            # Update workflow values if provided
            if fields:
                for path, value in fields.items():
                    # Parse path components
                    parts = path.split('.')
                    if not parts:
                        log.warning(f"Invalid field path: {path}")
                        continue

                    # Find the target node
                    node_result = self._find_node_by_title_or_id(workflow, parts[0])
                    if not node_result:
                        log.warning(f"Node not found: {parts[0]}")
                        continue
                        
                    node_id, node = node_result
                    
                    # Get node title for image naming
                    node_title = node.get('_meta', {}).get('title', '').strip()
                    if not node_title:
                        node_title = f"node_{node_id}"
                    
                    # Ensure inputs object exists
                    if 'inputs' not in node:
                        node['inputs'] = {}

                    # Handle special cases for different value types
                    if is_image(value) or (isinstance(value, str) and any(value.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])):
                        # Upload any image input to the server
                        try:
                            # Always enable overwrite and use consistent naming
                            response = await self.upload_image(
                                value,
                                folder_type="input",
                                overwrite=True,
                                filename=f"INPUT_{node_title}"
                            )
                            value = {
                                "filename": response["name"],
                                "subfolder": response.get("subfolder", ""),
                                "type": response.get("type", "input")
                            }
                            pass
                        except Exception as e:
                            log.error(f"Failed to upload image for {path}: {e}")
                            continue

                    # Handle different path formats
                    if len(parts) == 1 or (len(parts) == 2 and parts[1] == 'value'):
                        # Use first input field
                        field_name = self._get_default_input_name(node)
                        if not field_name:
                            log.warning(f"No input fields found for node: {parts[0]}")
                            continue
                        node['inputs'][field_name] = value
                        log.info(f"{node_title}.inputs.{field_name} → {value}")
                        
                    elif len(parts) == 2:
                        # Direct field name
                        node['inputs'][parts[1]] = value
                        log.info(f"{node_title}.inputs.{parts[1]} → {value}")
                        
                    elif len(parts) == 3 and parts[1] == 'inputs':
                        # Full path format
                        node['inputs'][parts[2]] = value
                        log.info(f"{node_title}.inputs.{parts[2]} → {value}")
                    else:
                        log.warning(f"Invalid field path format: {path}")
                        continue

            # Post workflow to ComfyUI
            log.info("[cyan]Executing workflow via API...[/cyan]")
            
            # Queue the prompt using direct API endpoint
            ret = await self._make_request(
                "POST", 
                "/prompt", 
                {"prompt": workflow, "client_id": self.client_id}
            )
            
            if not wait:
                return cast(WorkflowResponse, ret)
                
            if not isinstance(ret, dict) or "prompt_id" not in ret:
                raise ValueError(f"Invalid response format: {ret}")
                
            prompt_id = ret["prompt_id"]
            
            # Wait for execution
            await asyncio.sleep(1) # minimum time to prevent race condition
            await self.await_execution(require_uiapi=False)
            
            # Get execution results
            history = await self.get_history_async(prompt_id)
            if not isinstance(history, dict) or prompt_id not in history:
                raise ValueError(f"Invalid history response: {history}")
                
            history_data = cast(HistoryData, history[prompt_id])
            
            # Process outputs
            outputs: Dict[str, list[np.ndarray]] = {}
            for node_id, node_output in history_data.get("outputs", {}).items():
                if "images" in node_output:
                    images = []
                    for image_info in node_output["images"]:
                        image_data = await self.get_image_async(
                            image_info["filename"],
                            image_info["subfolder"],
                            image_info["type"]
                        )
                        if isinstance(image_data, np.ndarray):
                            images.append(image_data)
                    outputs[node_id] = images
                    
            return {
                "prompt_id": prompt_id,
                "outputs": outputs,
                "history": history_data
            }
            
        except Exception as e:
            log.error(f"[red]Workflow execution failed:[/red] {str(e)}")
            raise

    def execute_workflow(self, workflow: dict, wait: bool = True) -> Dict[str, Any]:
        """Sync version of execute_workflow_async"""
        return self.run_sync(self.execute_workflow_async(workflow, wait))

    # UTILITIES
    # ----------------------------------------

    def check_webui_connection_status(self) -> dict:
        """Check if ComfyUI web interface is connected and get status info"""
        try:
            return self.run_sync(self.json_get_async("/uiapi/connection_status"))
        except Exception as e:
            log.error(f"Error checking connection status: {e}")
            return {"status": "error", "error": str(e), "webui_connected": False}

    @property
    def is_webui_ready(self) -> bool:
        """Check if WebUI is ready without waiting"""
        return self._webui_ready

    def set_webui_disconnected(self):
        """Called when connection is lost"""
        self._webui_ready = False
        log.warning("[bold red]WebUI Disconnected[/bold red]")

    def set_webui_check_interval(self, interval: float):
        """Set the interval between WebUI availability checks"""
        self._webui_check_interval = max(0.1, interval)  # Minimum 100ms

    @staticmethod
    def find_output_node(json_object) -> Optional[str]:
        """Find the SaveImage node in the workflow"""
        for key, value in json_object.items():
            if isinstance(value, dict):
                if value.get("class_type") in ["SaveImage", "Image Save"]:
                    return key
                result = ComfyClient.find_output_node(value)
                if result:
                    return result
        return None

    async def upload_image(
        self, 
        image_input: Union[str, Image.Image, np.ndarray], 
        subfolder: Optional[str] = None, 
        folder_type: Optional[str] = None, 
        overwrite: bool = False,
        filename: Optional[str] = None
    ) -> UploadResponse:
        """Upload an image to the ComfyUI server.
        
        Args:
            image_input: One of:
                - Path to an image file (str)
                - PIL Image object
                - Numpy array (HWC with 3 channels, RGB)
            subfolder: Optional subfolder to store the image in
            folder_type: Optional type of folder (input, temp, output)
            overwrite: Whether to overwrite existing files
            filename: Optional filename to use (without extension)
            
        Returns:
            Dict containing the server's response with keys:
            - name: filename of uploaded image
            - subfolder: subfolder where image was stored
            - type: type of folder (input, temp, output)
        """
        try:
            url = f"/upload/image"
            
            # Prepare multipart form data
            data = {'overwrite': str(overwrite).lower()}
            if subfolder:
                data['subfolder'] = subfolder
            if folder_type:
                data['type'] = folder_type
                
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                with open(image_input, 'rb') as f:
                    files = {'image': (f"{filename}.png" if filename else None, f, 'image/png')}
                    response = await self._make_request("POST", url, data=data, files=files)
            elif isinstance(image_input, np.ndarray):
                # Numpy array (HWC, RGB)
                if image_input.dtype in [np.float32, np.float64]:
                    image_input = (image_input * 255).astype(np.uint8)
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    # Convert RGB to BGR for cv2
                    image_input = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
                # Encode to PNG
                success, buffer = cv2.imencode(".png", image_input)
                if not success:
                    raise ValueError("Failed to encode image")
                # Create file-like object
                img_byte_arr = io.BytesIO(buffer.tobytes())
                files = {'image': (f"{filename}.png" if filename else "image.png", img_byte_arr, 'image/png')}
                response = await self._make_request("POST", url, data=data, files=files)
            else:
                # PIL Image
                img_byte_arr = io.BytesIO()
                image_input.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                files = {'image': (f"{filename}.png" if filename else "image.png", img_byte_arr, 'image/png')}
                response = await self._make_request("POST", url, data=data, files=files)
                
            if not isinstance(response, dict):
                raise ValueError(f"Invalid upload response: {response}")
                
            return cast(UploadResponse, response)
                
        except Exception as e:
            log.error(f"[red]Failed to upload image:[/red] {str(e)}")
            raise
