# read version from installed package
from importlib.metadata import PackageNotFoundError, version

from .probounds import (
    calculate_bounds_combined,
    calculate_bounds_combined_by_feature,
    calculate_bounds_experimental,
    calculate_bounds_experimental_from_probounds_data,
    calculate_bounds_observed,
    calculate_bounds_observed_by_feature,
    calculate_bounds_observed_from_probounds_data,
    create_probounds_crosstab,
    probounds_crosstab_feature,
)

try:  # pragma: no cover - fall back if package isn't installed
    __version__ = version("probounds")
except PackageNotFoundError:  # pragma: no cover - allow tests to run without install
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "calculate_bounds_combined",
    "calculate_bounds_combined_by_feature",
    "calculate_bounds_experimental",
    "calculate_bounds_experimental_from_probounds_data",
    "calculate_bounds_observed",
    "calculate_bounds_observed_by_feature",
    "calculate_bounds_observed_from_probounds_data",
    "create_probounds_crosstab",
    "probounds_crosstab_feature",
]
