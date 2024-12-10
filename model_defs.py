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

from huggingface_hub import hf_hub_download
import gdown
from .civitai import download_file

log = logging.getLogger(__name__)

class ControlNetType(Enum):
    STANDARD = auto()
    XS = auto()

@dataclass
class ModelDef:
    """Represents paths for a specific model type with fallback options"""
    huggingface: Optional[str]
    local: Optional[str]
    civitai: Optional[str]
    gdrive: Optional[str]
    fp16: bool = False
    ckpt_type: str = "checkpoints"

    def __init__(self,
                 huggingface: Optional[str] = None,
                 local: Optional[str] = None,
                 civitai: Optional[str] = None,
                 gdrive: Optional[str] = None,
                 fp16: bool = True,
                 ckpt_type: str = "checkpoints"):
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
            "ckpt_type": self.ckpt_type
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
        if repo_id.startswith('https://huggingface.co/'):
            # Remove prefix and split path
            parts = repo_id[len('https://huggingface.co/'):].split('/')
            if len(parts) >= 2:
                # Take just org/repo, ignore the rest
                repo_id = '/'.join(parts[:2])
        return repo_id

    def resolve_local_path(self, type) -> str:
        """
        Resolves model path, checking local first then falling back to huggingface
        Returns: (path, is_local)
        """
        if self.local:
            local_path = get_model_path(type, self.local)
            if local_path and os.path.exists(local_path):
                log.info(f"Resolved model: {local_path}")
                return local_path

        if self.huggingface:
            log.info(f"Resolved {self.huggingface}")
            return self.huggingface

        if self.civitai:
            raise NotImplementedError("Civitai not yet!!")

        raise ValueError("No valid model path found")

    def download_huggingface(self, path: str) -> str:
        """Download model from Hugging Face and return local path"""
        if not self.huggingface:
            raise ValueError("No Hugging Face URL provided")
        
        assert self.huggingface_id

        # Extract just the filename if it's a direct file link
        if any(self.huggingface.endswith(ext) for ext in ['.safetensors', '.bin', '.ckpt']):
            final_path = determine_model_path(self.ckpt_type, path)

            log.info(f"Downloading from HuggingFace")
            log.info(f"from: {self.huggingface}")
            log.info(f"to: {final_path}")

            if has_model(self.ckpt_type, path):
                log.info(f"Model already exists at {final_path}!")
                return final_path.as_posix()

            filename = self.huggingface.split('/')[-1]
            downloaded_path = hf_hub_download(self.huggingface_id, filename=filename)
            downloaded_path = Path(downloaded_path).resolve()

            os.makedirs(final_path.parent, exist_ok=True)
            log.info(f"Moving {Path(downloaded_path).resolve()} to {final_path.as_posix()}")

            downloaded_path.rename(final_path)

            return final_path.as_posix()
        
        raise ValueError("Only direct file links supported for HuggingFace downloads")
    
    def read_civitai_token(self) -> Optional[str]:
        paths = [
            Path.home().joinpath('.civitai_token').resolve().as_posix(),
            (Path(__file__).parent.parent / ".civitai_token").as_posix()
        ]
        for path in paths:  
            if os.path.exists(path):
                return open(path).read().strip()
        return None

    def download_civitai(self, path: str) -> str:
        """Download model from CivitAI and return local path"""
        from . import civitai

        if not self.civitai:
            raise ValueError("No CivitAI URL provided")

        # Download using CivitAI helper
        try:
            token = self.read_civitai_token()

            final_path = determine_model_path(self.ckpt_type, path)
            if has_model(self.ckpt_type, path):
                log.info(f"Model already on disk!")
                return final_path.as_posix()
            
            log.info(f"Downloading from CivitAI")
            log.info(f"from: {self.civitai}")
            log.info(f"to: {final_path}")
            
            # if has_model(self.ckpt_type(), path):
            #     log.info(f"Model already on disk!")
            #     return final_path.as_posix()

            downloaded_path = civitai.download_file(url=self.civitai, 
                                                    output_file=final_path.as_posix(), 
                                                    token=token)

            os.makedirs(final_path.parent, exist_ok=True)
            shutil.move(downloaded_path, final_path)

            return final_path.as_posix()
        except Exception as e:
            log.info(f"Error downloading from CivitAI: {str(e)}")
            raise
    
    def download_gdrive(self, path: str) -> str:
        """Download model from Google Drive and return local path"""
        if not self.gdrive:
            raise ValueError("No Google Drive URL provided")

        final_path = determine_model_path(self.ckpt_type, path)
        if has_model(self.ckpt_type, path):
            log.info(f"Model already on disk!")
            return final_path.as_posix()

        log.info(f"Downloading from Google Drive")
        log.info(f"from: {self.gdrive}")
        log.info(f"to: {final_path}")

        # Extract file ID from Google Drive URL
        file_id = None
        if '/file/d/' in self.gdrive:
            file_id = self.gdrive.split('/file/d/')[1].split('/')[0]
        elif 'id=' in self.gdrive:
            file_id = self.gdrive.split('id=')[1].split('&')[0]
        
        if not file_id:
            raise ValueError("Could not extract Google Drive file ID from URL")

        os.makedirs(final_path.parent, exist_ok=True)
        
        # Download using gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        downloaded_path = gdown.download(url, final_path.as_posix(), quiet=False)
        
        if downloaded_path is None:
            raise RuntimeError("Failed to download file from Google Drive")

        return final_path.as_posix()

    def download(self, path: str):
        if self.civitai:
            return self.download_civitai(path)
        elif self.huggingface:
            return self.download_huggingface(path)
        elif self.gdrive:
            return self.download_gdrive(path)
        else:
            raise ValueError("No valid model path found")

@dataclass
class ControlNetDef(ModelDef):
    """Represents a ControlNet configuration with path and type information"""
    type: ControlNetType = ControlNetType.STANDARD

    def __init__(self,
                 huggingface: Optional[str] = None,
                 local: Optional[str] = None,
                 civitai: Optional[str] = None,
                 gdrive: Optional[str] = None,
                 fp16: bool = True):
        super().__init__(huggingface, local, civitai, gdrive, fp16)
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
        if ("fp16" in (self.huggingface or "")
            or "fp16" in (self.local or "")
            or "fp16" in (self.civitai or "")
            or self.fp16):
            return "fp16"
        else:
            return None

@dataclass
class LoraDef(ModelDef):
    """Represents a LoRA configuration with path and strength information"""
    unet_strength: float = 1.0
    text_encoder_strength: Optional[float] = None  # If None, uses unet_strength
    fuse: bool = False

    def __init__(self,
                 huggingface: Optional[str] = None,
                 local: Optional[str] = None,
                 civitai: Optional[str] = None,
                 gdrive: Optional[str] = None,
                 weights: float | tuple[float, float] = 1.0,
                 fuse=True):
        super().__init__(huggingface, local, civitai, gdrive)
        self.unet_strength = weights[0] if isinstance(weights, tuple) else weights
        self.text_encoder_strength = weights[1] if isinstance(weights, tuple) else weights
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
            log.info("CivitAI API support not yet implemented")
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
        return self.original_path.split('#')[0]

    def __repr__(self):
        return f"LoRASource({self.original_path})"

def get_model_path(type, ckpt_name, required=False):
    """
    Get a SD model full path from its name, searching all the defined model locations
    """
    import folder_paths
    ret = folder_paths.get_full_path(type, ckpt_name)
    if ret is not None:
        return ret

    if required:
        raise ValueError(f"Model {ckpt_name} not found")
    return None

def determine_model_path(type, ckpt_name) -> Path:
    import folder_paths
    root = folder_paths.folder_names_and_paths[type][0][0]
    return Path(root) / ckpt_name

def has_model(type, ckpt_name) -> bool:
    return get_model_path(type, ckpt_name) is not None
