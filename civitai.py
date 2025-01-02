#!/usr/bin/env python3
import logging
import os.path
import re
import sys
import argparse
import time
from typing import Optional
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install

# ANSI color codes
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m" 
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
RESET = "\033[0m"

# Text style codes
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
ITALIC = "\033[3m"

# Log prefixes
LOG_DOWNLOAD = f"{BLUE}(download){RESET}"
LOG_PROGRESS = f"{GREEN}(progress){RESET}"
LOG_ERROR = f"{RED}(error){RESET}"
LOG_TOKEN = f"{MAGENTA}(token){RESET}"

# Set up rich console and logging
console = Console()
install(show_locals=True)

# Configure logging with rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_time=True, show_path=True)]
)

log = logging.getLogger("civitai")

# Remove any existing handlers to avoid duplicate logging
for handler in log.handlers[:]:
    log.removeHandler(handler)

CHUNK_SIZE = 1638400
TOKEN_FILE = Path.home() / '.civitai' / 'config'
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
API_DOWNLOAD_BASE = "https://civitai.com/api/download/models"
API_MODEL_BASE = "https://civitai.com/api/v1/models"

def get_args():
    parser = argparse.ArgumentParser(
        description='CivitAI Downloader',
    )

    parser.add_argument(
        'url',
        type=str,
        help='CivitAI Download URL, eg: https://civitai.com/api/download/models/46846'
    )

    parser.add_argument(
        'output_path',
        type=str,
        help='Output path, eg: /workspace/stable-diffusion-webui/models/Stable-diffusion'
    )

    return parser.parse_args()


def get_token() -> Optional[str]:
    try:
        with open(TOKEN_FILE, 'r') as file:
            token = file.read()
            log.info(f"{LOG_TOKEN} Loaded token from {TOKEN_FILE}")
            return token
    except Exception as e:
        log.warning(f"{LOG_TOKEN} No token found at {TOKEN_FILE}")
        return None


def store_token(token: str):
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_FILE, 'w') as file:
        file.write(token)
    log.info(f"{LOG_TOKEN} Stored token at {TOKEN_FILE}")


def prompt_for_civitai_token():
    token = input('Please enter your CivitAI API token: ')
    store_token(token)
    log.info(f"{LOG_TOKEN} New token stored")
    return token

def convert_url_to_api_download_url(url: str) -> str:
    """Convert a civitai.com model page URL to download URL"""
    try:
        if url.startswith("urn"):
            # example: urn:air:sdxl:lora:civitai:144510@160616
            # we need to extract the model_id and version_id from the URL
            model_id, version_id = url.split(":")[-1].split("@")
            log.debug(f"{LOG_DOWNLOAD} Converting URN to download URL: {model_id}@{version_id}")
            return f"{API_DOWNLOAD_BASE}/{version_id}"
        elif url.startswith("https://civitai.com/models/"):
            # case 1: https://civitai.com/models/144510/collagexl
            # case 2: https://civitai.com/models/139562?modelVersionId=798204
            import re

            # Extract model ID from URL
            model_match = re.search(r'/models/(\d+)', url)
            if not model_match:
                raise ValueError(f"Could not extract model ID from URL: {url}")
            model_id = model_match.group(1)
            
            # Try to extract version ID from query param if present
            version_match = re.search(r'modelVersionId=(\d+)', url)
            version_id = version_match.group(1) if version_match else model_id

            log.debug(f"{LOG_DOWNLOAD} Converting model URL to download URL: {model_id}@{version_id}")
            if version_id:
                return f"{API_DOWNLOAD_BASE}/{version_id}"
            else:
                return f"{API_DOWNLOAD_BASE}/{model_id}"
        elif url.startswith("https://civitai.com/api/download/models/"):
            # case 3: https://civitai.com/api/download/models/144510/160616
            # already in the correct format
            log.debug(f"{LOG_DOWNLOAD} URL already in download format")
            return url
        else:
            raise ValueError(f"Could not convert CivitAI URL to download URL: {url}")
    except Exception as e:
        log.error(f"{LOG_ERROR} Failed to convert URL: {e}")
        raise


def download_file(url: str, 
                  output_dir: Optional[str] = None, 
                  output_name: Optional[str] = None, 
                  output_file: Optional[str] = None,
                  token: Optional[str] = None) -> str:
    token = token or get_token()
    if not token:
        token = prompt_for_civitai_token()
        
    url = convert_url_to_api_download_url(url)
    log.info(f"{LOG_DOWNLOAD} Starting download from {url}")

    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': USER_AGENT,
    }

    # Disable automatic redirect handling
    class NoRedirection(urllib.request.HTTPErrorProcessor):
        def http_response(self, request, response):
            return response
        https_response = http_response

    request = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(NoRedirection)
    response = opener.open(request)

    if response.status in [301, 302, 303, 307, 308]:
        redirect_url = response.getheader('Location')
        log.debug(f"{LOG_DOWNLOAD} Following redirect to {redirect_url}")

        # Extract filename from the redirect URL
        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        content_disposition = query_params.get('response-content-disposition', [None])[0]

        if content_disposition:
            response_filename = unquote(content_disposition.split('filename=')[1].strip('"'))
            log.debug(f"{LOG_DOWNLOAD} Extracted filename: {response_filename}")
        else:
            log.error(f"{LOG_ERROR} No filename in content disposition")
            raise Exception('Unable to determine filename')

        response = urllib.request.urlopen(redirect_url)
    elif response.status == 404:
        log.error(f"{LOG_ERROR} File not found at {url}")
        raise Exception('File not found')
    else:
        log.error(f"{LOG_ERROR} No redirect found for {url}")
        raise Exception('No redirect found, something went wrong')

    total_size = response.getheader('Content-Length')

    if total_size is not None:
        total_size = int(total_size)
        log.info(f"{LOG_DOWNLOAD} Total size: {total_size / (1024*1024):.1f}MB")

    if not output_file:
        assert output_dir, "output_dir is required if output_file is not provided"
        output_file = os.path.join(output_dir, output_name or response_filename)

    log.info(f"{LOG_DOWNLOAD} Downloading to {output_file}")
    with open(output_file, 'wb') as f:
        downloaded = 0
        start_time = time.time()
        last_progress = -1

        while True:
            chunk_start_time = time.time()
            buffer = response.read(CHUNK_SIZE)
            chunk_end_time = time.time()

            if not buffer:
                break

            downloaded += len(buffer)
            f.write(buffer)
            chunk_time = chunk_end_time - chunk_start_time

            if chunk_time > 0:
                speed = len(buffer) / chunk_time / (1024 ** 2)  # Speed in MB/s

            if total_size is not None:
                progress = downloaded / total_size
                progress_10pct = int(progress * 10)  # Convert to 10% increments
                
                if progress_10pct > last_progress:
                    last_progress = progress_10pct
                    log.info(f"{LOG_PROGRESS} {output_file} [{progress*100:.0f}%] - {speed:.2f} MB/s")

    end_time = time.time()
    time_taken = end_time - start_time
    hours, remainder = divmod(time_taken, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        time_str = f'{int(hours)}h {int(minutes)}m {int(seconds)}s'
    elif minutes > 0:
        time_str = f'{int(minutes)}m {int(seconds)}s'
    else:
        time_str = f'{int(seconds)}s'

    log.info(f"{LOG_DOWNLOAD} {BOLD}Completed in {time_str}{RESET}")
    return output_file


