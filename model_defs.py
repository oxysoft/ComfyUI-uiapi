from abc import abstractmethod
import logging
from multiprocessing.util import abstract_sockets_supported
import os
from dataclasses import dataclass
from enum import auto, Enum
from pathlib import Path
import shutil
from typing import Optional, Tuple
from urllib.parse import urlparse
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install

from huggingface_hub import hf_hub_download
import gdown
from .civitai import download_file

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
LOG_MODEL = f"{GREEN}(model){RESET}"
LOG_PATH = f"{BLUE}(path){RESET}"
LOG_DOWNLOAD = f"{MAGENTA}(download){RESET}"
LOG_ERROR = f"{RED}(error){RESET}"
LOG_HF = f"{CYAN}(huggingface){RESET}"
LOG_CIVITAI = f"{YELLOW}(civitai){RESET}"
LOG_GDRIVE = f"{BLUE}(gdrive){RESET}"

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

log = logging.getLogger("model_defs")

# Remove any existing handlers to avoid duplicate logging
for handler in log.handlers[:]:
    log.removeHandler(handler)


class ControlNetType(Enum):
    STANDARD = auto()
    XS = auto()


@dataclass
class ModelDef:
    """Represents paths for a specific model type with fallback options"""

    url: Optional[str]
    huggingface: Optional[str]
    local: Optional[str]
    civitai: Optional[str]
    gdrive: Optional[str]
    fp16: bool = False
    ckpt_type: str = "checkpoints"

    def __init__(
        self,
        url: Optional[str] = None,
        huggingface: Optional[str] = None,
        local: Optional[str] = None,
        civitai: Optional[str] = None,
        gdrive: Optional[str] = None,
        fp16: bool = True,
        ckpt_type: str = "checkpoints",
    ):
        self.url = url
        self.huggingface = huggingface
        self.local = local
        self.civitai = civitai
        self.gdrive = gdrive
        self.fp16 = fp16
        self.ckpt_type = ckpt_type

    def to_dict(self) -> dict:
        """Convert ModelDef to a dictionary for JSON serialization"""
        return {
            "huggingface": self.huggingface,
            "local": self.local,
            "civitai": self.civitai,
            "gdrive": self.gdrive,
            "fp16": self.fp16,
            "ckpt_type": self.ckpt_type,
        }

    # if auto:
    #     # Auto-detect source type from URL format
    #     if 'civitai' in auto:
    #         self.civitai = auto
    #     elif 'huggingface' in auto:
    #         self.huggingface = auto
    #     else:
    #         self.local = auto

    @property
    def huggingface_id(self) -> Optional[str]:
        """Extract organization/model repo ID from HuggingFace URL or direct ID"""
        if not self.huggingface:
            return None

        repo_id = self.huggingface
        if repo_id.startswith("https://huggingface.co/"):
            # Remove prefix and split path
            parts = repo_id[len("https://huggingface.co/") :].split("/")
            if len(parts) >= 2:
                # Take just org/repo, ignore the rest
                repo_id = "/".join(parts[:2])
        return repo_id

    def resolve_local_path(self, type) -> str:
        """
        Resolves model path, checking local first then falling back to huggingface
        Returns: (path, is_local)
        """
        if self.local:
            local_path = get_model_path(self.local, type)
            if local_path and os.path.exists(local_path):
                log.info(f"{LOG_PATH} Resolved local model: {BOLD}{local_path}{RESET}")
                return local_path

        if self.huggingface:
            log.info(f"{LOG_HF} Using HuggingFace model: {BOLD}{self.huggingface}{RESET}")
            return self.huggingface

        if self.civitai:
            raise NotImplementedError("Civitai not yet!!")

        raise ValueError("No valid model path found")

    def _get_final_path(self, path: str) -> Path:
        """Get the final path for the model and create parent directories"""
        final_path = det_model_path(path, self.ckpt_type)
        os.makedirs(final_path.parent, exist_ok=True)
        return final_path

    def _log_download_start(self, source: str, url: str, final_path: Path):
        """Log the start of a download operation"""
        log_prefix = {
            "HuggingFace": LOG_HF,
            "CivitAI": LOG_CIVITAI,
            "Google Drive": LOG_GDRIVE
        }.get(source, LOG_DOWNLOAD)
        
        log.info(f"{log_prefix} Starting download from {source}")
        log.info(f"{log_prefix} Source: {BOLD}{url}{RESET}")
        log.info(f"{log_prefix} Target: {BOLD}{final_path}{RESET}")

    def _download_huggingface(self, final_path: Path, url: str, id: str) -> Path:
        """Core HuggingFace download logic"""

        if not any(url.endswith(ext) for ext in [".safetensors", ".bin", ".ckpt"]):
            raise ValueError(
                "Only direct file links supported for HuggingFace downloads"
            )

        filename = url.split("/")[-1]
        downloaded_path = Path(hf_hub_download(id, filename=filename)).resolve()

        log.info(f"{LOG_HF} Moving {BOLD}{downloaded_path}{RESET} to {BOLD}{final_path.as_posix()}{RESET}")
        downloaded_path.rename(final_path)
        return final_path

    def _download_civitai(self, final_path: Path, url: str) -> Path:
        """Core CivitAI download logic"""
        from . import civitai

        token = self.read_civitai_token()
        downloaded_path = civitai.download_file(
            url=url, output_file=final_path.as_posix(), token=token
        )
        log.info(f"{LOG_CIVITAI} Moving {BOLD}{downloaded_path}{RESET} to {BOLD}{final_path}{RESET}")
        shutil.move(downloaded_path, final_path)
        return final_path

    def _download_gdrive(self, final_path: Path, url: str) -> Path:
        """Core Google Drive download logic"""
        # Extract file ID from Google Drive URL
        file_id = None
        if "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
        elif "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]

        if not file_id:
            log.error(f"{LOG_ERROR} Could not extract Google Drive file ID from URL: {url}")
            raise ValueError("Could not extract Google Drive file ID from URL")

        # Download using gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        downloaded_path = gdown.download(url, final_path.as_posix(), quiet=False)

        if downloaded_path is None:
            log.error(f"{LOG_ERROR} Failed to download file from Google Drive")
            raise RuntimeError("Failed to download file from Google Drive")

        return final_path

    def download(
        self, ckpt: str, source: Optional[str] = None, url: Optional[str] = None
    ) -> str:
        """
        Download model from configured source and return local path.

        Args:
            path: The path/name for the downloaded model
            source: Optional source override ('civitai', 'huggingface', or 'gdrive')
                   If not specified, tries sources in order: CivitAI -> HuggingFace -> Google Drive

        Returns:
            str: Path where the model was saved
        """
        # Check if model already exists
        final_path = self._get_final_path(ckpt)
        if has_model(ckpt, self.ckpt_type):
            log.info(f"{LOG_MODEL} Model {BOLD}{ckpt}{RESET} already exists at {final_path}")
            return final_path.as_posix()

        # Handle generic URL-based model definition
        url = url or self.url
        civitai = self.civitai
        huggingface = self.huggingface
        gdrive = self.gdrive
        if url and not civitai and not huggingface and not gdrive:
            if "civitai" in url:
                source = "civitai"
                civitai = url
            elif "huggingface" in url:
                source = "huggingface"
                huggingface = url
            elif "drive" in url:
                source = "gdrive"
                gdrive = url
            else:
                log.error(f"{LOG_ERROR} Invalid model URL: {url}")
                raise ValueError(f"Invalid model URL: {url}")

        try:
            # Use specified source if provided
            if source:
                if source == "civitai" and civitai:
                    self._log_download_start("CivitAI", civitai, final_path)
                    final_path = self._download_civitai(final_path, civitai)
                elif source == "huggingface" and huggingface:
                    if not self.huggingface_id:
                        log.error(f"{LOG_ERROR} Invalid Hugging Face URL/ID")
                        raise ValueError("Invalid Hugging Face URL/ID")
                    self._log_download_start("HuggingFace", huggingface, final_path)
                    final_path = self._download_huggingface(
                        final_path, huggingface, self.huggingface_id
                    )
                elif source == "gdrive" and gdrive:
                    self._log_download_start("Google Drive", gdrive, final_path)
                    final_path = self._download_gdrive(final_path, gdrive)
                else:
                    log.error(f"{LOG_ERROR} Source '{source}' not available for this model")
                    raise ValueError(f"Source '{source}' not available for this model")

            # Otherwise try sources in default order
            else:
                if civitai:
                    self._log_download_start("CivitAI", civitai, final_path)
                    final_path = self._download_civitai(final_path, civitai)
                elif huggingface:
                    if not self.huggingface_id:
                        log.error(f"{LOG_ERROR} Invalid Hugging Face URL/ID")
                        raise ValueError("Invalid Hugging Face URL/ID")
                    self._log_download_start("HuggingFace", huggingface, final_path)
                    final_path = self._download_huggingface(
                        final_path, huggingface, self.huggingface_id
                    )
                elif gdrive:
                    self._log_download_start("Google Drive", gdrive, final_path)
                    final_path = self._download_gdrive(final_path, gdrive)
                else:
                    log.error(f"{LOG_ERROR} No valid model path found")
                    raise ValueError("No valid model path found")

        except Exception as e:
            log.error(f"{LOG_ERROR} Error downloading model: {str(e)}")
            raise

        log.info(f"{LOG_MODEL} Successfully downloaded to {BOLD}{final_path.as_posix()}{RESET}")
        return final_path.as_posix()

    def read_civitai_token(self) -> Optional[str]:
        paths = [
            Path.home().joinpath(".civitai_token").resolve().as_posix(),
            (Path(__file__).parent.parent / ".civitai_token").as_posix(),
        ]
        for path in paths:
            if os.path.exists(path):
                return open(path).read().strip()
        return None


@dataclass
class ControlNetDef(ModelDef):
    """Represents a ControlNet configuration with path and type information"""

    type: ControlNetType = ControlNetType.STANDARD

    def __init__(
        self,
        url: Optional[str] = None,
        huggingface: Optional[str] = None,
        local: Optional[str] = None,
        civitai: Optional[str] = None,
        gdrive: Optional[str] = None,
        fp16: bool = True,
    ):
        super().__init__(url, huggingface, local, civitai, gdrive, fp16)
        self.ckpt_type = "controlnet"

    def is_xs(self):
        return self.type == ControlNetType.XS

    def get_classtype(self):
        from diffusers.models.controlnet import ControlNetModel

        if self.is_xs():
            # return ControlNetXSAdapter
            raise NotImplementedError("ControlNetXSAdapter not yet implemented")
        else:
            return ControlNetModel

    def get_variant(self):
        if (
            "fp16" in (self.huggingface or "")
            or "fp16" in (self.local or "")
            or "fp16" in (self.civitai or "")
            or self.fp16
        ):
            return "fp16"
        else:
            return None


class VaeDef(ModelDef):
    def __init__(self, path: str):
        super().__init__(path)
        self.ckpt_type = "vae"


@dataclass
class LoraDef(ModelDef):
    """Represents a LoRA configuration with path and strength information"""

    unet_strength: float = 1.0
    text_encoder_strength: Optional[float] = None  # If None, uses unet_strength
    fuse: bool = False

    def __init__(
        self,
        url: Optional[str] = None,
        huggingface: Optional[str] = None,
        local: Optional[str] = None,
        civitai: Optional[str] = None,
        gdrive: Optional[str] = None,
        weights: float | tuple[float, float] = 1.0,
        fuse=True,
    ):
        super().__init__(url, huggingface, local, civitai, gdrive)
        self.unet_strength = weights[0] if isinstance(weights, tuple) else weights
        self.text_encoder_strength = (
            weights[1] if isinstance(weights, tuple) else weights
        )
        self.fuse = fuse
        self.ckpt_type = "loras"


class LoRASource:
    """Represents a source location for a LoRA file with priority handling"""

    def __init__(self, path: str):
        self.original_path = path
        self.path: Path | str = Path(path) if not self._is_url(path) else path

    def _is_url(self, path: str) -> bool:
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False

    @property
    def is_local(self) -> bool:
        return not self._is_url(self.original_path)

    @property
    def exists(self) -> bool:
        return self.is_local and Path(self.path).exists()

    @property
    def is_civitai(self) -> bool:
        return "civitai.com" in str(self.path)

    @property
    def is_gdrive(self) -> bool:
        return "drive.google.com" in str(self.path)

    @property
    def civitai_url(self) -> Optional[str]:
        """Returns API download URL if this is a CivitAI source"""
        if not self.is_civitai:
            return None

        from . import civitai

        try:
            return civitai.convert_url_to_api_download_url(str(self.path))
        except NotImplementedError:
            log.info(f"{LOG_CIVITAI} API support not yet implemented")
            return None

    @property
    def is_air_tag(self) -> bool:
        """Check if the source is an AIR tag"""
        return self.original_path.startswith("urn:air:")

    @property
    def air_tag(self) -> Optional[str]:
        """Extract the AIR tag, ignoring any additional text after '#'"""
        if not self.is_air_tag:
            return None
        return self.original_path.split("#")[0]

    def __repr__(self):
        return f"LoRASource({self.original_path})"


def get_model_path(ckpt_name, type=None, required=False):
    """
    Get a SD model full path from its name, searching all the defined model locations.
    If type is None, searches all model types.
    """
    import folder_paths

    if type is not None:
        ret = folder_paths.get_full_path(type, ckpt_name)
        if ret is not None:
            log.debug(f"{LOG_PATH} Found {BOLD}{ckpt_name}{RESET} in {type} at {BOLD}{ret}{RESET}")
            return ret
    else:
        # Try all model types if type is None
        for model_type in folder_paths.folder_names_and_paths.keys():
            ret = folder_paths.get_full_path(model_type, ckpt_name)
            if ret is not None:
                log.debug(f"{LOG_PATH} Found {BOLD}{ckpt_name}{RESET} in {model_type} at {BOLD}{ret}{RESET}")
                return ret

    if required:
        log.error(f"{LOG_ERROR} Model {BOLD}{ckpt_name}{RESET} not found")
        raise ValueError(f"Model {ckpt_name} not found")
    return None


def has_model(ckpt_name, type=None) -> bool:
    return get_model_path(ckpt_name, type) is not None


def det_model_path(ckpt_name, type) -> Path:
    import folder_paths

    root = folder_paths.folder_names_and_paths[type][0][0]
    log.debug(f"{LOG_PATH} Determined path for {BOLD}{ckpt_name}{RESET} in {type}: {BOLD}{root}{RESET}")
    return Path(root) / ckpt_name
