import json
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


def load_irt_model(
    repo_id: str, 
    filename: str,
) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset"
    )
    irt_model = pd.read_csv(path, index_col=0)
    if "c" not in irt_model.columns:
        irt_model["c"] = 0
    if "d" not in irt_model.columns:
        irt_model["d"] = 1
    return irt_model


def load_lm_eval_results(
    repo_id: str,
    filename: str,
    binary: bool = True,
) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset"
    )
    eval_results = pd.read_csv(path, index_col=0)
    return eval_results.ge(0.5).astype(int) if binary else eval_results


def load_open_llm_leaderboard_results() -> Any:
    path = hf_hub_download(
        repo_id="allenai/fluid-benchmarking",
        repo_type="dataset",
        filename=f"data/open_llm_leaderboard_results.json",
    )
    with open(path, "r") as f:
        return json.load(f)
    

def load_id_to_item_map() -> Any:
    path = hf_hub_download(
        repo_id="allenai/fluid-benchmarking",
        repo_type="dataset",
        filename=f"data/id_to_item_map.json",
    )
    with open(path, "r") as f:
        return json.load(f)
