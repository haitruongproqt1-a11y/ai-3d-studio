# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import logging
import os
from functools import wraps

import torch


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = get_logger('hy3dgen.shapgen')


class synchronize_timer:
    """ Synchronized timer to count the inference time of `nn.Module.forward`.

        Supports both context manager and decorator usage.

        Example as context manager:
        ```python
        with synchronize_timer('name') as t:
            run()
        ```

        Example as decorator:
        ```python
        @synchronize_timer('Export to trimesh')
        def export_to_trimesh(mesh_output):
            pass
        ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        """Context manager entry: start timing."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Context manager exit: stop timing and log results."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                logger.info(f'{self.name} takes {self.time} ms')

    def __call__(self, func):
        """Decorator: wrap the function to time its execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper


def smart_load_model(
    model_path,
    subfolder,
    use_safetensors,
    variant,
):
    original_model_path = model_path
    base_dir = os.environ.get('HY3DGEN_MODELS', os.path.expanduser('~/.cache/hy3dgen'))
    local_dir = os.path.join(base_dir, model_path.replace('/', os.sep), subfolder.replace('/', os.sep))

    extension = 'ckpt' if not use_safetensors else 'safetensors'
    variant_str = '' if variant is None else f'.{variant}'
    ckpt_name = f'model{variant_str}.{extension}'

    config_path = os.path.join(local_dir, 'config.yaml')
    ckpt_path = os.path.join(local_dir, ckpt_name)

    if os.path.exists(config_path) and os.path.exists(ckpt_path):
        logger.info(f'Loaded model from local cache: {local_dir}')
        return config_path, ckpt_path

    logger.info(f'Model not found locally at {local_dir}, downloading from HuggingFace...')
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(local_dir, exist_ok=True)
        config_file = f"{subfolder}/config.yaml"
        ckpt_file = f"{subfolder}/{ckpt_name}"
        
        logger.info(f"Downloading {config_file}...")
        hf_config = hf_hub_download(
            repo_id=original_model_path,
            filename=config_file,
            local_dir=os.path.join(base_dir, model_path.replace('/', os.sep)),
        )
        logger.info(f"Downloading {ckpt_file}...")
        hf_ckpt = hf_hub_download(
            repo_id=original_model_path,
            filename=ckpt_file,
            local_dir=os.path.join(base_dir, model_path.replace('/', os.sep)),
        )
        return hf_config, hf_ckpt
    except ImportError:
        logger.warning("huggingface_hub is required to download models.")
        raise RuntimeError(f"Model path {model_path} not found")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise e

