from functools import lru_cache
from pathlib import Path
import os


ROOT_DIR_ENV = "PINN_SWE_ROOT_DIR"
DATA_DIR_ENV = "PINN_SWE_DATA_DIR"
TRAINING_RESULTS_DIR_ENV = "PINN_SWE_TRAINING_RESULTS_DIR"


def _validate_dir(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")
    return path


@lru_cache(maxsize=1)
def get_root_dir() -> Path:
    env_root = os.getenv(ROOT_DIR_ENV)
    if env_root:
        return _validate_dir(Path(env_root), ROOT_DIR_ENV)

    start = Path(__file__).resolve()

    for candidate in [start, *start.parents]:
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src" / "pinn_swe_reflections").is_dir()
        ):
            return candidate

    raise RuntimeError(
        f"Cannot find project root. Set {ROOT_DIR_ENV}=/path/to/project"
    )


def _resolve_from_env_or_default(env_name: str, default: Path) -> Path:
    env_path = os.getenv(env_name)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return default.expanduser().resolve()


ROOT_DIR = get_root_dir()

DATA_DIR = _resolve_from_env_or_default(
    DATA_DIR_ENV,
    ROOT_DIR / "Numerical_Solution" / "dt=1s_dx=400m",
)

TRAINING_RESULTS_DIR = _resolve_from_env_or_default(
    TRAINING_RESULTS_DIR_ENV,
    ROOT_DIR / "training_results",
)


def get_numerical_solution_dir(case: str) -> Path:
    return DATA_DIR / case


def get_training_run_dir(experiment_name: str) -> Path:
    return TRAINING_RESULTS_DIR / experiment_name