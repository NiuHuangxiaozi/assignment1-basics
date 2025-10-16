import importlib.metadata
from importlib import metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except metadata.PackageNotFoundError:
    # 当包还没被“安装”（例如在开发阶段、没有 dist-info）时，提供一个 fallback
    __version__ = "0.0.0"

