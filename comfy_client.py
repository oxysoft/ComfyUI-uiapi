import asyncio
import json
import logging
import math
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image
from websocket._core import WebSocket
from websocket._exceptions import WebSocketTimeoutException

from .model_defs import ModelDef

import base64
import io

from rich.logging import RichHandler
from rich.console import Console

from http.client import RemoteDisconnected
import backoff  # Add to requirements.txt

import threading

# Set up rich console
console = Console()

# Configure logging with rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=True
    )]
)

log = logging.getLogger("comfy_client")

# Allows using uiapi plugin to directly control the comfy webui
# ------------------------------------------------------------
def is_image(value):
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
        success, buffer = cv2.imencode('.png', image)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer.tobytes()).decode()
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

class ComfyClient:
    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = None  # Initialize to None, will be set up in ensure_connection
        self._webui_ready = False
        self._connection_lock = asyncio.Lock()
        self._reconnect_backoff = 1.0  # Initial backoff in seconds
        self._sync_lock = threading.Lock()  # Add thread safety for sync operations
        log.info(f"""[bold green]ComfyClient initialized[/bold green]
            Server: {server_address}
            Client ID: {self.client_id}""")

    async def ensure_connection(self):
        """Ensure both HTTP and WebSocket connections are alive"""
        async with self._connection_lock:
            while True:
                try:
                    # Check/establish HTTP connection
                    response = await self._make_request_once('GET', '/uiapi/connection_status')
                    if not response.get('webui_connected'):
                        raise ConnectionError("WebUI not connected")

                    # Check/establish WebSocket connection
                    if not (self.ws and self.ws.connected):
                        log.info("[yellow]Establishing WebSocket connection...[/yellow]")
                        self.ws = WebSocket()
                        self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
                        self.ws.timeout = 1

                    self._webui_ready = True
                    return True

                except Exception as e:
                    self._webui_ready = False
                    wait_time = min(self._reconnect_backoff * 2, 30)
                    log.warning(f"""[yellow]Connection attempt failed:[/yellow]
                        Error: {str(e)}
                        Retrying in {wait_time:.1f}s...""")
                    await asyncio.sleep(wait_time)
                    self._reconnect_backoff = wait_time

    async def _make_request_once(self, method: str, url: str, data: dict = None, verbose: bool = False) -> dict:
        """Single attempt at making a request without retry logic"""
        address = f'{self.server_address}{url}'
        if not address.startswith('http'):
            address = f'http://{address}'

        if verbose:
            log.debug(f"[cyan]Request details:[/cyan]\n" + 
                    f"  URL: {address}\n" +
                    f"  Method: {method}\n" +
                    f"  Data: {data if data else 'None'}")

        headers = {'Content-Type': 'application/json'}
        if data is not None:
            data = json.dumps(data).encode('utf-8')
            
        req = urllib.request.Request(address, headers=headers, data=data, method=method)
        with urllib.request.urlopen(req) as response:
            ret = json.loads(response.read())
            if verbose:
                log.info(f"[green]Request successful:[/green] {method} {url}")
                log.debug(f"Response: {ret}")
            return ret

    async def _make_request(self, method: str, url: str, data: dict = None, verbose: bool = False) -> dict:
        """Make request with automatic reconnection"""
        while True:
            try:
                await self.ensure_connection()
                return await self._make_request_once(method, url, data, verbose)
            except Exception as e:
                log.error(f"""[red]Request failed:[/red]
                    Method: {method}
                    URL: {url}
                    Error: {str(e)}
                    Retrying...""")
                self._webui_ready = False
                continue

    @classmethod
    def ConnectNew(cls, server_address: str, timeout: Optional[float] = None) -> 'ComfyClient':
        """Synchronous connection method"""
        # Create event loop if there isn't one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(cls.ConnectNewAsync(server_address, timeout))

    @classmethod
    async def ConnectNewAsync(cls, server_address: str, timeout: Optional[float] = None) -> 'ComfyClient':
        """Async connection method"""
        client = cls(server_address)
        try:
            start_time = time.time()
            while True:
                try:
                    await client.ensure_connection()
                    return client
                except Exception as e:
                    if timeout and (time.time() - start_time) > timeout:
                        raise TimeoutError(f"Connection timeout after {timeout}s")
                    continue
        except Exception as e:
            log.error(f"[bold red]Failed to establish connection:[/bold red] {str(e)}")
            raise

    def __getattr__(self, name):
        """Handle attribute lookup for async/sync methods"""
        try:
            attr = super().__getattribute__(name)
        except AttributeError:
            # Check if this is a sync version of an async method
            async_name = f"{name}_async"
            if hasattr(self, async_name):
                async_attr = getattr(self, async_name)
                if asyncio.iscoroutinefunction(async_attr):
                    return self._make_sync_wrapper(async_attr)
            raise
        
        # Convert async methods to sync if called directly
        if asyncio.iscoroutinefunction(attr):
            return self._make_sync_wrapper(attr)
        return attr

    def _make_sync_wrapper(self, async_func):
        """Create a sync version of an async function"""
        def sync_wrapper(*args, **kwargs):
            with self._sync_lock:  # Thread safety for sync operations
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(async_func(*args, **kwargs))
        return sync_wrapper

    async def await_execution(self):
        """Wait for execution with automatic reconnection"""
        start_time = time.time()
        max_timeout = 90
        last_status_time = 0

        while True:
            try:
                await self.ensure_connection()  # This handles both HTTP and WS connections
                
                out = self.ws.recv()
                current_time = time.time()
                elapsed = current_time - start_time

                if current_time - last_status_time >= 5:
                    log.info(f"[cyan]Execution status:[/cyan] Running for {elapsed:.1f}s")
                    last_status_time = current_time

                if isinstance(out, str):
                    msg = json.loads(out)
                    if msg['type'] == 'status' and msg['data']['status']['exec_info']['queue_remaining'] == 0:
                        log.info(f"[green]Execution completed[/green] in {elapsed:.1f}s")
                        return

            except Exception as e:
                log.warning(f"""[yellow]Execution monitoring interrupted:[/yellow]
                    Error: {str(e)}
                    Elapsed: {elapsed:.1f}s
                    Attempting to restore connection...""")
                self._webui_ready = False
                continue

            if time.time() - start_time > max_timeout:
                raise TimeoutError(f"Execution timeout after {max_timeout}s")

    def free(self):
        self.freed = True

    async def get_workflow(self):
        """Get current workflow state"""
        return await self.json_post('/uiapi/get_workflow')

    async def query_fields(self, verbose=False):
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

        response = self.json_post('/uiapi/query_fields', verbose=verbose)

        log.info(f"query_fields response: {response}")
        log.info(response)

        if isinstance(response, dict):
            if 'result' in response:
                return response['result']
            elif 'response' in response:
                return response['response']
            else:
                raise ValueError("Unexpected response format from server")
        else:
            raise ValueError("Unexpected response format from server")


    async def get_field(self, path_or_paths, verbose=False):
        """Get field values with proper async handling"""
        is_single = isinstance(path_or_paths, str)
        paths = [path_or_paths] if is_single else path_or_paths

        try:
            response = await self.json_post('/uiapi/get_fields', {
                "fields": paths,
            }, verbose=verbose)

            if isinstance(response, dict):
                if 'result' in response:
                    result = response['result']
                    return result[path_or_paths] if is_single else result
                elif 'response' in response:
                    result = response['response']
                    return result[path_or_paths] if is_single else result
                
            raise ValueError(f"Unexpected response format: {response}")
        except Exception as e:
            log.error(f"[red]Failed to get field values:[/red] {str(e)}")
            raise


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


    async def set(self, path_or_fields, value=None, verbose=False, clamp=False):
        """Set field values with proper async handling"""
        try:
            fields = [(path_or_fields, value)] if isinstance(path_or_fields, str) else path_or_fields
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
                        processed_fields.append([path, {
                            "type": "image_base64",
                            "data": base64_img
                        }])
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
            return await self.json_post('/uiapi/set_fields', {
                "fields": processed_fields,
            }, verbose=verbose)
        except Exception as e:
            log.error(f"[red]Failed to set fields:[/red] {str(e)}")
            raise


    def connect(self, path1, path2, verbose=False):
        return self.json_post('/uiapi/set_connection', {
            "field": [path1, path2],
        }, verbose=verbose)


    async def execute(self, wait=True):
        """
        Execute the workflow and return the image of the final SaveImage node
        """
        try:
            await self.ensure_connection()
            
            log.info("[cyan]Executing prompt...[/cyan]")
            ret = await self.json_post('/uiapi/execute')
            
            if not wait:
                return ret

            if not isinstance(ret, dict):
                raise ValueError(f"Unexpected response format: {ret}")

            # Handle broken responses with retry
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                if 'response' not in ret or 'prompt_id' not in ret.get('response', {}):
                    retry_count += 1
                    log.warning(f"[yellow]Broken response (attempt {retry_count}/{max_retries}), retrying...[/yellow]")
                    log.debug(f"Response was: {ret}")
                    await asyncio.sleep(1)  # Brief delay before retry
                    ret = await self.json_post('/uiapi/execute')
                else:
                    break
            
            if retry_count == max_retries:
                raise ValueError("Maximum retries exceeded for broken responses")

            exec_id = ret['response']['prompt_id']
            await self.await_execution()

            log.info("[green]Execution completed, fetching results...[/green]")

            workflow_json = await self.get_workflow()
            if not isinstance(workflow_json, dict) or 'response' not in workflow_json:
                raise ValueError(f"Invalid workflow response: {workflow_json}")

            output_node_id = self.find_output_node(workflow_json['response'])
            if not output_node_id:
                raise ValueError("No output node found in workflow")

            history = await self.get_history(exec_id)
            history_data = history[exec_id]

            filenames = history_data['outputs'][output_node_id]['images']
            if not filenames:
                log.warning("[yellow]No images found in execution output[/yellow]")
                return None

            info = filenames[0]
            return await self.get_image(info['filename'], info['subfolder'], info['type'])

        except Exception as e:
            log.error(f"[red]Execution failed:[/red] {str(e)}")
            raise

    async def get_history(self, prompt_id):
        """Get execution history with retry logic"""
        return await self._make_request('GET', f'/history/{prompt_id}')

    async def get_image(self, filename, subfolder, folder_type):
        """Get image with retry logic"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        
        try:
            response = await self._make_request('GET', f'/view?{url_values}')
            image = cv2.imdecode(np.frombuffer(response, np.uint8), cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            log.error(f"[red]Failed to get image:[/red] {str(e)}")
            raise

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

    async def json_post(self, url: str, input_data: dict = None, verbose: bool = False) -> dict:
        """Make POST request with retry logic"""
        data = input_data or {}
        if isinstance(data, str):
            data = json.loads(data)
        
        data['verbose'] = verbose
        data['client_id'] = self.client_id
        
        return await self._make_request('POST', url, data, verbose)

    async def json_get(self, url: str, verbose: bool = False) -> dict:
        """Make GET request with retry logic"""
        return await self._make_request('GET', url, None, verbose)

    def set_webui_disconnected(self):
        """Called when connection is lost"""
        self._webui_ready = False
        log.warning("[bold red]WebUI Disconnected[/bold red]")

    def initialize_ws(self):
        """Initialize WebSocket connection with error handling"""
        try:
            client_id = str(uuid.uuid4())
            self.ws = WebSocket()
            self.ws.connect(f"ws://{self.server_address}/ws?clientId={client_id}")
            self.ws.timeout = 1
            self.freed = False
            log.info("[green]WebSocket connection initialized[/green]")
        except Exception as e:
            self._webui_ready = False
            log.error(f"[red]Failed to initialize WebSocket:[/red] {str(e)}")
            raise ConnectionError(f"WebSocket initialization failed: {str(e)}")

    def check_connection_status(self) -> dict:
        """Check if ComfyUI web interface is connected and get status info"""
        try:
            return self.json_get('/uiapi/connection_status')
        except Exception as e:
            log.error(f"Error checking connection status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'webui_connected': False
            }

    def set_webui_check_interval(self, interval: float):
        """Set the interval between WebUI availability checks"""
        self._webui_check_interval = max(0.1, interval)  # Minimum 100ms

    @property
    def is_webui_ready(self) -> bool:
        """Check if WebUI is ready without waiting"""
        return self._webui_ready

    async def download_models_async(self, models: Dict[str, ModelDef], timeout: int = 300) -> Dict[str, Dict[str, Any]]:
        """
        Download models with automatic retry and progress tracking.
        """
        download_table = {
            filename: model_def.to_dict()
            for filename, model_def in models.items()
        }

        log.info(f"""[cyan]Starting model downloads:[/cyan]
            Models to download: {len(models)}
            Timeout per model: {timeout}s""")

        try:
            # Start the download process
            response = await self._make_request('POST', '/uiapi/download_models', {
                "download_table": download_table
            })

            if not isinstance(response, dict):
                raise ValueError(f"Unexpected response format: {response}")

            # Get download ID from response
            download_id = response.get('download_id')
            if not download_id:
                raise ValueError("No download_id in response")

            # Track download progress
            start_time = time.time()
            last_status_time = 0
            
            while True:
                try:
                    # Use the download status endpoint
                    status = await self._make_request('GET', f'/uiapi/download_status/{download_id}')
                    current_time = time.time()
                    elapsed = current_time - start_time

                    # Log status every 5 seconds
                    if current_time - last_status_time >= 5:
                        progress = status.get('progress', {})
                        completed = sum(1 for info in progress.values() if info.get('status') == 'success')
                        total = len(models)
                        log.info(f"""[cyan]Download status:[/cyan]
                            Progress: {completed}/{total} models
                            Elapsed time: {elapsed:.1f}s""")
                        last_status_time = current_time

                    # Check if downloads are complete
                    if status.get('completed', False):
                        log.info(f"[green]All downloads completed[/green] in {elapsed:.1f}s")
                        return status.get('progress', {})

                    # Check for any errors
                    for model, info in status.get('progress', {}).items():
                        if info.get('status') == 'error':
                            log.error(f"[red]Error downloading {model}:[/red] {info.get('error')}")

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

    def download_models(self, models: Dict[str, ModelDef], timeout: int = 300):
        """Sync version that calls async version"""
        return self.__getattr__('download_models_async')(models, timeout)

    async def download_model_async(self, model_name: str, model_def: ModelDef, timeout: int = 300) -> Dict[str, Any]:
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

            response = await self._make_request('POST', '/uiapi/download_model', {
                "model_name": model_name,
                "model_def": model_def.to_dict()
            })

            if not isinstance(response, dict):
                raise ValueError(f"Unexpected response format: {response}")

            # Monitor download progress
            start_time = time.time()
            last_status_time = 0
            
            while True:
                try:
                    status = await self._make_request('GET', f'/uiapi/model_status/{model_name}')
                    current_time = time.time()
                    elapsed = current_time - start_time

                    # Log status every 5 seconds
                    if current_time - last_status_time >= 5:
                        log.info(f"""[cyan]Download status for {model_name}:[/cyan]
                            Status: {status.get('status', 'unknown')}
                            Elapsed time: {elapsed:.1f}s""")
                        last_status_time = current_time

                    if status.get('status') == 'success':
                        log.info(f"[green]Download completed[/green] in {elapsed:.1f}s")
                        return status

                    if status.get('status') == 'error':
                        error = status.get('error', 'Unknown error')
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

    def download_model(self, model_name: str, model_def: ModelDef, timeout: int = 300) -> Dict[str, Any]:
        """Sync version of direct model download"""
        return self.__getattr__('download_model_async')(model_name, model_def, timeout)

    async def get_model_status_async(self, model_name: str) -> Dict[str, Any]:
        """
        Get status of a specific model.
        
        Args:
            model_name: Name of the model file to check
            
        Returns:
            Dict containing model status information
        """
        try:
            return await self._make_request('GET', f'/uiapi/model_status/{model_name}')
        except Exception as e:
            log.error(f"[red]Failed to get model status for {model_name}:[/red] {str(e)}")
            raise

    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Sync version of model status check"""
        return self.__getattr__('get_model_status_async')(model_name)
