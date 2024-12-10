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

    def _get_final_path(self, path: str) -> Path:
        """Get the final path for the model and create parent directories"""
        final_path = determine_model_path(self.ckpt_type, path)
        os.makedirs(final_path.parent, exist_ok=True)
        return final_path

    def _log_download_start(self, source: str, url: str, final_path: Path):
        """Log the start of a download operation"""
        log.info(f"Downloading from {source}")
        log.info(f"from: {url}")
        log.info(f"to: {final_path}")

    def _download_huggingface(self, final_path: Path) -> Path:
        """Core HuggingFace download logic"""
        assert self.huggingface is not None
        assert self.huggingface_id is not None

        if not any(self.huggingface.endswith(ext) for ext in ['.safetensors', '.bin', '.ckpt']):
            raise ValueError("Only direct file links supported for HuggingFace downloads")

        filename = self.huggingface.split('/')[-1]
        downloaded_path = Path(hf_hub_download(self.huggingface_id, filename=filename)).resolve()
        
        log.info(f"Moving {downloaded_path} to {final_path.as_posix()}")
        downloaded_path.rename(final_path)
        return final_path

    def _download_civitai(self, final_path: Path) -> Path:
        """Core CivitAI download logic"""
        from . import civitai
        assert self.civitai is not None
        token = self.read_civitai_token()
        
        downloaded_path = civitai.download_file(
            url=self.civitai,
            output_file=final_path.as_posix(),
            token=token
        )
        shutil.move(downloaded_path, final_path)
        return final_path

    def _download_gdrive(self, final_path: Path) -> Path:
        """Core Google Drive download logic"""
        # Extract file ID from Google Drive URL
        assert self.gdrive is not None
        file_id = None
        if '/file/d/' in self.gdrive:
            file_id = self.gdrive.split('/file/d/')[1].split('/')[0]
        elif 'id=' in self.gdrive:
            file_id = self.gdrive.split('id=')[1].split('&')[0]
        
        if not file_id:
            raise ValueError("Could not extract Google Drive file ID from URL")
        
        # Download using gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        downloaded_path = gdown.download(url, final_path.as_posix(), quiet=False)
        
        if downloaded_path is None:
            raise RuntimeError("Failed to download file from Google Drive")
        
        return final_path

    def download(self, path: str, source: Optional[str] = None) -> str:
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
        final_path = self._get_final_path(path)
        if has_model(self.ckpt_type, path):
            log.info(f"Model {path} already exists at {final_path}!")
            return final_path.as_posix()

        try:
            # Use specified source if provided
            if source:
                if source == 'civitai' and self.civitai:
                    self._log_download_start("CivitAI", self.civitai, final_path)
                    final_path = self._download_civitai(final_path)
                elif source == 'huggingface' and self.huggingface:
                    if not self.huggingface_id:
                        raise ValueError("Invalid Hugging Face URL/ID")
                    self._log_download_start("HuggingFace", self.huggingface, final_path)
                    final_path = self._download_huggingface(final_path)
                elif source == 'gdrive' and self.gdrive:
                    self._log_download_start("Google Drive", self.gdrive, final_path)
                    final_path = self._download_gdrive(final_path)
                else:
                    raise ValueError(f"Source '{source}' not available for this model")
            
            # Otherwise try sources in default order
            else:
                if self.civitai:
                    self._log_download_start("CivitAI", self.civitai, final_path)
                    final_path = self._download_civitai(final_path)
                elif self.huggingface:
                    if not self.huggingface_id:
                        raise ValueError("Invalid Hugging Face URL/ID")
                    self._log_download_start("HuggingFace", self.huggingface, final_path)
                    final_path = self._download_huggingface(final_path)
                elif self.gdrive:
                    self._log_download_start("Google Drive", self.gdrive, final_path)
                    final_path = self._download_gdrive(final_path)
                else:
                    raise ValueError("No valid model path found")

        except Exception as e:
            log.error(f"Error downloading model: {str(e)}")
            raise

        return final_path.as_posix()

    def read_civitai_token(self) -> Optional[str]:
        paths = [
            Path.home().joinpath('.civitai_token').resolve().as_posix(),
            (Path(__file__).parent.parent / ".civitai_token").as_posix()
        ]
        for path in paths:  
            if os.path.exists(path):
                return open(path).read().strip()
        return None

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
