import math
import shutil
from pathlib import Path

from ..graph_state import GraphState
from .get_excel_batch_node import read_openalex_excel_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_TRAIN_PATH = PROJECT_ROOT / "data" / "matscholar" / "train.txt"
DEFAULT_WORK_TRAIN_PATH = (
    PROJECT_ROOT / "data" / "GPT_data_source" / "add_train_graph" / "train.txt"
)
DEFAULT_VALID_PATH = PROJECT_ROOT / "data" / "matscholar" / "valid.txt"
DEFAULT_EXCEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "GPT_data_source"
    / "excel_data"
    / "openalex_materials_abstracts_500.xlsx"
)


def _resolve_required_file(path_value: str | Path, field_name: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist or is not a file: {path}")
    return path


def _copy_base_train_pool(target_path: Path) -> None:
    source_path = _resolve_required_file(BASE_TRAIN_PATH, "base_train_path")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def initialize_add_train_node(graph_state: GraphState) -> dict:
    """
    Initialize add_train_graph from the full MatScholar train split and an
    OpenAlex Excel pool.
    """
    train_path = Path(graph_state.train_path or DEFAULT_WORK_TRAIN_PATH).expanduser()
    valid_path = Path(graph_state.valid_path or DEFAULT_VALID_PATH).expanduser()
    excel_path = Path(graph_state.unlabeled_pool_path or DEFAULT_EXCEL_PATH).expanduser()

    paper_batch_size = int(graph_state.paper_batch_size or 100)
    if paper_batch_size <= 0:
        raise ValueError("paper_batch_size must be a positive integer.")

    _copy_base_train_pool(train_path)
    _resolve_required_file(train_path, "train_path")
    _resolve_required_file(valid_path, "valid_path")
    _resolve_required_file(excel_path, "unlabeled_pool_path")

    total_paper_count = len(read_openalex_excel_records(excel_path))
    iterations = math.ceil(total_paper_count / paper_batch_size) + 1

    return {
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "unlabeled_pool_path": str(excel_path),
        "paper_batch_size": paper_batch_size,
        "total_paper_count": total_paper_count,
        "iterations": max(1, iterations),
        "iteration": 0,
        "distance_ratio_threshold": 0.50,
        "min_distance_ratio_threshold": 0.30,
        "max_distance_ratio_threshold": 0.80,
        "model_distance_ratio_threshold": 0.30,
        "threshold_step": 0.05,
        "current_batch": {},
        "processed_sample_ids": [],
        "processed_paper_ids": [],
        "ner_bio_results": {},
        "ner_entity_dicts": {},
        "llm_outputs": {},
        "model_entity_dicts": {},
        "distance_ratio_records": {},
        "decision_records": {},
        "previous_metrics": [],
        "best_metrics": {
            "loss": float("inf"),
            "precision": -1.0,
            "recall": -1.0,
            "f1": -1.0,
        },
    }
