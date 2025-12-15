from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import ast
import math
import warnings
import functools

import cv2
import numpy as np
import torch

def is_ascii(s) -> bool:
    """Check if a string is composed of only ASCII characters.

    Args:
        s (str | list | tuple | dict): Input to be checked (all are converted to string for checking).

    Returns:
        (bool): True if the string is composed only of ASCII characters, False otherwise.
    """
    return all(ord(c) < 128 for c in str(s))

@functools.lru_cache
def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str): Name to be used in warning message.
        hard (bool): If True, raise an AssertionError if the requirement is not met.
        verbose (bool): If True, print warning message if requirement is not met.
        msg (str): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Examples:
        Check if current version is exactly 22.04
        >>> check_version(current="22.04", required="==22.04")

        Check if current version is greater than or equal to 22.04
        >>> check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        Check if current version is less than or equal to 22.04
        >>> check_version(current="22.04", required="<=22.04")

        Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        >>> check_version(current="21.10", required=">20.04,<22.04")
    """
    if not current:  # if current is '' or None
        LOGGER.warning(f"invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(f"{current} package is required but not installed") from e
            else:
                return False

    if not required:  # if required is '' or None
        return True

    if "sys_platform" in required and (  # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        if not op:
            op = ">="  # assume >= if no op passed
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and not (c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"{name}{required} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(warning)  # assert version requirements met
        if verbose:
            LOGGER.warning(warning)
    return result


def check_latest_pypi_version(package_name="ultralytics"):
    """Return the latest version of a PyPI package without downloading or installing it.

    Args:
        package_name (str): The name of the package to find the latest version for.

    Returns:
        (str): The latest version of the package.
    """
    import requests  # scoped as slow import

    try:
        requests.packages.urllib3.disable_warnings()  # Disable the InsecureRequestWarning
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=3)
        if response.status_code == 200:
            return response.json()["info"]["version"]
    except Exception:
        return None

def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the stride, update
    it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int| list[int]): Image size
        stride (int): Stride value
        min_dim (int): Minimum number of dimensions
        max_dim (int): Maximum number of dimensions
        floor (int): Minimum allowed value for image size
    Returns:
        (list[int] | int): Updated image size
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int): imgsz=[imgsz]
    elif isinstance(imgsz, (tuple,list)): imgsz=list(imgsz)
    elif isinstance(imgsz, str): imgsz=[int(imgsz)] if imgsz.isnumeric() else ast.literal_eval(imgsz)
    else:
        raise TypeError(f'imgsz={imgsz} is of invalid type {type(imgsz).__name__}'
                        f'Valid imgsz types are int `imgsz=640` or list `imgsz=[640,640]`')

    # Apply max_dim
    if len(imgsz)>max_dim:
        msg=("'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list"
             "or an integer, i.e., 'yolo export imgsz=640,480' or 'yolo export imgsz=640'")
        if max_dim!=1: raise ValueError(f'imgsz={imgsz} is not a valid image size. {msg}')
        warnings.warn(f'updating to "imgsz={max(imgsz)}". {msg}')
        imgsz=[max(imgsz)]

    # Make image size a multiple of the stride
    sz=[max(math.ceil(x/stride)*stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz!=imgsz: warnings.warn(f'imgsz={imgsz} must be multiple of max stride {stride}, updating it to {sz}')

    # Add missing dimensions if necessary
    sz=[sz[0], sz[0]] if min_dim==2 and len(sz)==1 else sz[0] if min_dim==1 and len(sz)==1 else sz

    return sz
    
        