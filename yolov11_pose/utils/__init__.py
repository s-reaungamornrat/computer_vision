import os
import yaml
import platform

from types import SimpleNamespace
from pathlib import Path

DEFAULT_CFG_PATH='../default.yaml'
if os.path.isfile(DEFAULT_CFG_PATH):
    with open(DEFAULT_CFG_PATH) as f: DEFAULT_CFG_DICT=yaml.load(f, Loader=yaml.SafeLoader)
else:
    DEFAULT_CFG_PATH='../../default.yaml'
    if os.path.isfile(DEFAULT_CFG_PATH):
        with open(DEFAULT_CFG_PATH) as f: DEFAULT_CFG_DICT=yaml.load(f, Loader=yaml.SafeLoader)

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
MACOS_VERSION = platform.mac_ver()[0] if MACOS else None
NOT_MACOS14 = not (MACOS and MACOS_VERSION.startswith("14."))

class IterableSimpleNamespace(SimpleNamespace):
    """An iterable SimpleNamespace class that provides enhanced functionality for attribute access and iteration.

    This class extends the SimpleNamespace class with additional methods for iteration, string representation, and
    attribute access. It is designed to be used as a convenient container for storing and accessing configuration
    parameters.

    Methods:
        __iter__: Return an iterator of key-value pairs from the namespace's attributes.
        __str__: Return a human-readable string representation of the object.
        __getattr__: Provide a custom attribute access error message with helpful information.
        get: Retrieve the value of a specified key, or a default value if the key doesn't exist.

    Examples:
        >>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
        >>> for k, v in cfg:
        ...     print(f"{k}: {v}")
        a: 1
        b: 2
        c: 3
        >>> print(cfg)
        a=1
        b=2
        c=3
        >>> cfg.get("b")
        2
        >>> cfg.get("d", "default")
        'default'

    Notes:
        This class is particularly useful for storing configuration parameters in a more accessible
        and iterable format compared to a standard dictionary.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Provide a custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


def is_dir_writeable(dir_path: str| Path)->bool:
    """Check if a directory is writable
    Args:
        dir_path (str| Path): The path to the directory
    Returns:
        (bool): True if the directory is writable, False otherwise
    """
    return os.access(str(dir_path), os.W_OK)