import time
import logging
import torch
from utility.utils_logger import logger
from typing import Any, Callable, Tuple, Optional

logging.basicConfig(level=logging.ERROR)

def get_device(device=None):
    if device is not None:
        return torch.device(device)

    # if none
    device_priority = {
        'cuda': [torch.cuda.is_available,
                 lambda x: logger.debug(f'Using CUDA device {torch.cuda.get_device_name(x)}')],
        'mps': [torch.backends.mps.is_available, lambda _: logger.debug('Using MPS device')],
        'cpu': [lambda: True, lambda x: logger.warning(
            f'You are running this script without CUDA or MPS (current device: {x}). It may be very slow')
                ]
    }

    for _device, (availability_check, log_msg) in device_priority.items():
        if availability_check():
            log_msg(_device)
            return torch.device(_device)

    raise Exception(f'Device {device if device else "any"} not available.')

# copy from kcg-ml-intrinsic-dimension/utility/utils.py
def execute_function(func: Callable, *args: Any, **kwargs: Any) -> Tuple[Optional[Any], Optional[str], float]:
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        error = None
    except Exception as e:
        logging.error("Error in function execution: %s", e)
        result = None
        error = str(e)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, error, elapsed_time

def measure_running_time(func: Callable, *args: Any, **kwargs: Any) -> Tuple[Optional[Any], float, Optional[str]]:
    timeout = 1  # seconds
    result, error, elapsed_time = execute_function(func, *args, **kwargs)

    if elapsed_time > timeout:
        logging.warning("Execution time exceeded 1 second. Retrying...")
        max_retries = 1
        for attempt in range(max_retries):
            result, error, elapsed_time = execute_function(func, *args, **kwargs)
            if elapsed_time <= timeout:
                break
            logging.warning(f"Retry {attempt + 1} exceeded 1 second.")

    return result, elapsed_time, error

def format_duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = ""
    formatted_time += f"{int(hours)}h " if hours > 0 else ""
    formatted_time += f"{int(minutes)}m " if minutes > 0 else ""
    formatted_time += f"{seconds:.3f}s " if seconds > 0 else "0s"
    return formatted_time