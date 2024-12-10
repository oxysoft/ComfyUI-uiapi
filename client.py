import io
import json
import logging
import re
import sys
import time
import traceback
import urllib.request
import uuid
from pathlib import Path
from urllib.error import HTTPError
import urllib.parse

import cv2
import numpy as np
from PIL import Image
from websocket._core import WebSocket
from websocket._exceptions import WebSocketTimeoutException

from typing import Dict, Any, Union, Optional
import math

from .model_defs import ModelDef

log = logging.getLogger('comfyui')

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

class ComfyClient:
    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = WebSocket()
        self.ws.timeout = 1
        self.freed = False
        self.last_imgs = dict()
    
    @classmethod
    def ConnectNew(cls, server_address: str):
        client = cls(server_address)
        client.ws.connect(f"ws://{server_address}/ws?clientId={client.client_id}")
        if client.ws.connected:
            log.info("Connected to ComfyUI server.")
        else:
            raise Exception("Failed to connect to ComfyUI server.")
        return client

    def initialize_ws(self):
        # First we connect to the server
        client_id = str(uuid.uuid4())
        # created_input_images = []
        self.ws = WebSocket()
        self.ws.connect(f"ws://{self.server_address}/ws?clientId={client_id}")
        self.ws.timeout = 1
        self.freed = False

        # Sleep for 5 seconds
        # time.sleep(5)
        # set_field('cfg', 0)
        # execute()
        # get_workflow()


    def free(self):
        global freed
        freed = True

    def get_workflow(self):
        return self.json_post('/uiapi/get_workflow')

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


    def get_field(self, path_or_paths, verbose=False):
        """
        Retrieve value(s) from specified node path(s).

        :param path_or_paths: String (single path) or List of strings (multiple paths)
        :param verbose: Boolean flag for verbose output
        :return: Single value if string input, Dictionary of path-value pairs if list input
        """
        is_single = isinstance(path_or_paths, str)
        paths = [path_or_paths] if is_single else path_or_paths

        response = self.json_post('/uiapi/get_fields', {
            "fields": paths,
        }, verbose=verbose)

        if isinstance(response, dict):
            if 'result' in response:
                result = response['result']
                return result[path_or_paths] if is_single else result
            elif 'response' in response:
                result = response['response']
                return result[path_or_paths] if is_single else result
            else:
                raise ValueError("Unexpected response format from server")
        else:
            raise ValueError("Unexpected response format from server")


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


    def set(self, path_or_fields, value=None, verbose=False, clamp=False):
        # Check if it's a single field or multiple fields
        if isinstance(path_or_fields, str):
            # Single field case
            fields = [(path_or_fields, value)]
        else:
            # Multiple fields case
            fields = path_or_fields

        processed_fields = []

        for path, val in fields:
            if not self.is_valid_value(val):
                log.warning(f"Skipping invalid value for {path}: {val}")
                continue

            if is_image(val):
                self.set_img(path, val, verbose)
            else:
                if clamp:
                    val = clamp01(val)
                processed_fields.append([path, val])

        if processed_fields:
            log.info(f"set_fields: {processed_fields}")
            return self.json_post('/uiapi/set_fields', {
                "fields": processed_fields,
            }, verbose=verbose)


    def connect(self, path1, path2, verbose=False):
        return self.json_post('/uiapi/set_connection', {
            "field": [path1, path2],
        }, verbose=verbose)



    def set_img(self, path, value, verbose=False):
        if path in self.last_imgs:
            Path(self.last_imgs[path]).unlink()

        # TODO we need to do this through a request
        input_name = f'INPUT_{path.split(".")[0]}.png'
        self.set(path, input_name, verbose)
        inputdir = Path(userconf.comfy_input_root)
        filename = inputdir / input_name

        self.last_imgs[path] = str(filename)
        if isinstance(value, Image.Image):
            value.save(filename)
        elif isinstance(value, np.ndarray):
            cv2.imwrite(str(filename), cv2.cvtColor(value, cv2.COLOR_RGB2BGR))


    def execute(self, wait=True):
        """
        Execute the workflow and return the image of the final SaveImage node
        """

        # if rv['steps'] < 1:
        #     set('ksampler.steps', 8)
        #     log.error("Steps was less than 1, setting to 8")

        while True:
            def get_history(prompt_id):  # This method is used to retrieve the history of a prompt from the API server
                with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
                    return json.loads(response.read())

            def get_image(filename, subfolder, folder_type):  # This method is used to retrieve an image from the API server
                data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
                url_values = urllib.parse.urlencode(data)
                with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
                    dat = response.read()
                    # image_file = io.BytesIO(dat)
                    # image = Image.open(image_file)
                    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    image = cv2.imdecode(np.frombuffer(dat, np.uint8), cv2.IMREAD_COLOR) # TODO this may not
                    return image

            def find_output_node(json_object) -> str:  # This method is used to find the node containing the SaveImage class in a prompt
                for key, value in json_object.items():
                    if isinstance(value, dict):
                        if value.get("class_type") == "SaveImage" or value.get("class_type") == "Image Save":
                            return key
                        result = find_output_node(value)
                        if result:
                            return result
                return None
            
            log.info("executing prompt")
            ret = self.json_post('/uiapi/execute')
            if not wait:
                break
            
            assert isinstance(ret, dict)

            exec_id = ret['response']['prompt_id']
            self.await_execution()

            workflow_json = self.get_workflow()
            output_node_id = find_output_node(workflow_json['response'])
            history = get_history(exec_id)[exec_id]

            filenames = history['outputs'][output_node_id]['images']

            if not filenames:
                return None

            info = filenames[0]
            filename = info['filename']
            subfolder = info['subfolder']
            folder_type = info['type']

            return get_image(filename, subfolder, folder_type)


    def await_execution(self):
        global ws
        global freed

        self.initialize_ws()

        max_timeout = 90
        starttime = time.time()
        while True:
            try:
                t1 = time.time()
                out = self.ws.recv()
                t2 = time.time()

                # if t2 - starttime < 0.5:
                #     # flushing old messages
                #     continue

                if isinstance(out, str):
                    msg = json.loads(out)
                    # log.info(f"await_executing: recv {msg}")
                    if msg['type'] == 'status' and msg['data']['status']['exec_info']['queue_remaining'] == 0:
                        log.info(f"await_executing: done")
                        return
                    else:
                        log.error(f'Unhandled message: {msg}')
            except (WebSocketTimeoutException, TimeoutError) as timeout_exc:
                pass
            except Exception as e:
                log.error(f"await_executing: error")
                log.error(e)
                return

            if freed:
                freed = False
                return


    def json_post(self, url, data=None, log_response=False, verbose=False)-> dict:
        data = data or {}
        if isinstance(data, str):
            data = json.loads(data)

        data['verbose'] = verbose
        data['client_id'] = self.client_id
        data = json.dumps(data)
        data = data.encode('utf-8')

        address = f'{self.server_address}{url}'
        if not address.startswith('http'):
            address = f'http://{address}'

        log.debug(f"POST {address} {data}")
        req = urllib.request.Request(
            address,
            headers={'Content-Type': 'application/json'},
            data=data, )

        # try:
        now = time.time()
        with urllib.request.urlopen(req) as response:
            log.debug(f"RECV {response}")
            ret = json.loads(response.read())
        elapsed_seconds = time.time() - now

        if log_response:
            log.info(f"POST {url} {data} -> {ret} (took {elapsed_seconds:.2f}s)")
        return ret
        # except HTTPError as e:
        #     log.error(str(e))
        #     traceback.print_exc()
        #     log.error("- Is the server running?")
        #     log.error("- Is the uiapi plugin OK?")
        #     return None

    def download_models(self, models: Dict[str, ModelDef]) -> Dict[str, Dict[str, Any]]:
        """
        Download models specified in the models dictionary.
        
        Args:
            models: A dictionary mapping model filenames to their ModelDef objects.
                Example:
                {
                    "model1.safetensors": ModelDef(
                        huggingface="org/repo/model1.safetensors",
                        fp16=True
                    ),
                    "model2.ckpt": ModelDef(
                        civitai="https://civitai.com/models/..."
                    )
                }
        
        Returns:
            A dictionary containing the download results for each model:
            {
                "model1.safetensors": {
                    "status": "success",
                    "path": "/path/to/downloaded/model"
                },
                "model2.ckpt": {
                    "status": "error",
                    "error": "Error message"
                }
            }
        """
        # Convert ModelDef objects to dictionaries
        download_table = {
            filename: model_def.to_dict() 
            for filename, model_def in models.items()
        }
        
        response = self.json_post('/uiapi/download_models', {
            "download_table": download_table
        })
        
        if isinstance(response, dict) and 'downloads' in response:
            return response['downloads']
        else:
            raise ValueError("Unexpected response format from server")


RENDERVAR_FIELD_MAPPING = {
    'prompt':     'prompt.text',
    'chg':        'ksampler.denoise',
    'steps':      'ksampler.steps',
    'seed':       'ksampler.seed',
    'cfg':        'ksampler.cfg',
    'sampler':    'ksampler.sampler_name',
    'img':        'img.image',
    'promptneg':  'promptneg.text',
    'ccg{}':      'ccg{}.strength',
    'ccg':        'ccg{}.strength',
    'guidance':   'cn_img{}.image',
    'seed_chg':   'noise.chg',
    'seed_grain': 'noise.grain',
}

RENDERVAR_CLAMP01_ARGS = ['chg']

# def mapping(rv):
#     return {
#         'ksampler.cfg':          rv.cfg,
#         'ksampler.steps':        rv.steps,
#         'ksampler.seed':         rv.seed,
#         'ksampler.denoise':      rv.chg,
#         'ksampler.sampler_name': rv.sampler,
#         'img':                   rv.img,
#         'prompt':                rv.prompt
#     }

# def txt2img(**args):
#     log.info("txt2img")
#     hud(diffusion='txt2img')
#     # args['chg'] = 0.925
#     set_values(args)
#     set('KSampler.positive', 1)
#     sconnect("prompt", "ccg1.CONDITIONING")
#     connect("ccg2.CONDITIONING", "ccg3.CONDITIONING")
#     connect("ccg3.CONDITIONING", "KSampler.positive")
#     return execute()


# def img2img(**args):
#     # TODO figure out if this unloads the controlnets (which we dont want, for max performance)
#     log.info("img2img")
#     hud(diffusion='img2img')
#     set_values(args)
#     connect("ConditioningAverage.CONDITIONING", "KSampler.positive")
#     connect("prompt", "KSampler.positive")
#     return execute()


# def set_rv(rv: Any) -> None:
#     set_values({field: getattr(rv, field) for field in RENDERVAR_FIELD_MAPPING.keys() if hasattr(rv, field)})


# def set_values(args: Dict[str, Union[Any, tuple]]) -> None:
#     batched_fields = []
#     for arg, value in args.items():
#         if not is_valid_value(value):
#             log.warning(f"Skipping invalid value for {arg}: {value}")
#             continue

#         with_clamp = arg in RENDERVAR_CLAMP01_ARGS
#         if with_clamp and not arg == 'cfg':  # TODO remove cfg
#             value = clamp01(value)

#         if arg in RENDERVAR_FIELD_MAPPING:
#             fields = RENDERVAR_FIELD_MAPPING[arg]
#             if isinstance(fields, str):
#                 fields = [fields]

#             for field in fields:
#                 if '{' in field:
#                     # Handle numbered fields
#                     if isinstance(value, (tuple, list)):
#                         # Handle tuple values
#                         batched_fields.extend([
#                             (field.format(i), tuple_value)
#                             for i, tuple_value in enumerate(value, start=1)
#                             if is_valid_value(tuple_value)
#                         ])
#                     else:
#                         # Handle individual numbered fields
#                         base_name = re.sub(r'\{.*?\}', '', arg)
#                         numbered_args = [k for k in args if k.startswith(base_name) and k[len(base_name):].isdigit()]
#                         batched_fields.extend([
#                             (field.format(k[len(base_name):]), args[k])
#                             for k in numbered_args
#                             if is_valid_value(args[k])
#                         ])
#                 else:
#                     # Handle non-numbered fields
#                     batched_fields.append((field, value))

#     # Send all fields in a single batch
#     if batched_fields:
#         set(batched_fields)