import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from fluid_benchmarking import config, engine, estimators

# Type aliases
IRTModel = Union[np.ndarray, pd.DataFrame]
Idxes = Union[Sequence[int], np.ndarray, pd.Index]
Responses = Union[Sequence[int], Sequence[float], np.ndarray, pd.Series]
    

def full_accuracy(
    lm_responses: Responses,
) -> float:
    return np.mean(lm_responses)


def full_ability(
    lm_responses: Responses,
    irt_model: IRTModel,
    estimation_method: str = "map",
) -> float:
    lm_responses = np.array(lm_responses)
    irt_model = np.array(irt_model)
    return estimators.ability_estimate(lm_responses, irt_model, estimation_method)


def random_accuracy(
    lm_responses: Responses,
    sample_idxes: Idxes,
) -> float:
    lm_responses = np.array(lm_responses)
    sample_idxes = np.array(sample_idxes)
    return np.mean(lm_responses[sample_idxes])


def random_ability(
    lm_responses: Responses,
    irt_model: IRTModel,
    sample_idxes: Idxes,
    estimation_method: str = "map",
) -> float:
    lm_responses = np.array(lm_responses, dtype=float, copy=True)
    irt_model = np.array(irt_model)

    # Hide items not in random subset
    mask = np.isin(np.arange(len(lm_responses)), sample_idxes)
    lm_responses[~mask] = np.nan

    return estimators.ability_estimate(lm_responses, irt_model, estimation_method)


def fluid_benchmarking(
    lm_responses: Responses,
    irt_model: IRTModel,
    start_ability: float,
    n_max: int,
    estimation_method: str = "map",
) -> Tuple[List[float], List[int]]:
    lm_responses = np.array(lm_responses)
    irt_model = np.array(irt_model)
    eval_fb = engine.run_fluid_benchmarking(
        lm_responses=lm_responses,
        irt_model=irt_model,
        start_ability=start_ability,
        n_max=n_max,
        method=estimation_method,
    )
    return eval_fb["abilities_fb"], eval_fb["items_fb"]


def iterate_evals(
    lm_responses: Responses,
    methods: Sequence[str],
    irt_model: Optional[IRTModel] = None,
    estimation_method_irt: str = "map",
    samples_dict: Optional[Dict[int, np.ndarray]] = None,
    start_ability_fb: float = 0,
    seed: int = 0,
) -> Dict[str, Union[float, List[float], List[int]]]:

    if any(m in config.IRT_METHODS for m in methods):
        if irt_model is None:
            raise ValueError("irt_model is required for IRT-based methods.")

    # Sample items in case not specified
    if samples_dict is None:
        random.seed(seed)
        n_items = len(lm_responses)
        samples_dict = {}
        for n_samples in config.N_SAMPLES_LIST:
            if n_samples > n_items:
                raise ValueError(f"Number of samples={n_samples} > number of items={n_items}.")
            samples_dict[n_samples] = np.array(
                random.sample(range(n_items), n_samples)
            )
    
    output = {}

    # Full accuracy
    if "full_accuracy" in methods:
        output["full_accuracy"] = full_accuracy(
            lm_responses
        )

    # Full ability
    if "full_ability" in methods:
        output["full_ability"] = full_ability(
            lm_responses, 
            irt_model,
            estimation_method_irt
        )

    # Random accuracy
    if "random_accuracy" in methods:
        for n_samples in samples_dict:
            output[f"random_accuracy_{n_samples}"] = random_accuracy(
                lm_responses, 
                samples_dict[n_samples]
            )

    # Random ability
    if "random_ability" in methods:
        for n_samples in samples_dict:
            output[f"random_ability_{n_samples}"] = random_ability(
                lm_responses, 
                irt_model,
                samples_dict[n_samples],
                estimation_method_irt
            )

    # Fluid Benchmarking
    if "fluid_benchmarking" in methods:
        abilities_fb, items_fb = fluid_benchmarking(
            lm_responses,
            irt_model,
            start_ability_fb,
            max(samples_dict.keys()),
            estimation_method_irt,
        )
        output["abilities_fb"] = abilities_fb
        output["items_fb"] = items_fb

    return output
