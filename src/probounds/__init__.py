# read version from installed package
from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - fall back if package isn't installed
    __version__ = version("probounds")
except PackageNotFoundError:  # pragma: no cover - allow tests to run without install
    __version__ = "0.0.0"
