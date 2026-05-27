from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs import BaseConfig


def find_root() -> Path:
    """Find the root directory of the project, by climbing up the directory tree"""
    curr = Path.cwd().resolve()
    for parent in [curr] + list(curr.parents):
        if parent.name == "Build_an_LLM":
            return parent
    return curr


def create_folder_structure(PROJECT_ROOT: Path) -> tuple[Path, Path, Path]:
    """Create the directory structure for the project."""
    print("\nProject Root:", PROJECT_ROOT)

    DATA_DIR = PROJECT_ROOT / "data"
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    CHPTS_DIR = ARTIFACTS_DIR / "checkpoints"

    for d in [
        DATA_DIR,
        ARTIFACTS_DIR,
        ARTIFACTS_DIR / "meta-models",  # Store Meta-models here
        ARTIFACTS_DIR / "models",
        ARTIFACTS_DIR / "universal_tokenizer",
        CHPTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    return DATA_DIR, ARTIFACTS_DIR, CHPTS_DIR

