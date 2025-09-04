import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages

from fluid_benchmarking import config, rutils

catr = rpackages.importr("catR")

# Type aliases
Responses = Union[Sequence[int], Sequence[float], np.ndarray, pd.Series]
Idxes = Union[Sequence[int], np.ndarray, pd.Index]
RObj = Any
    

def full_accuracy(
    lm_responses: Responses,
) -> float:
    return np.mean(lm_responses)


def full_ability(
    lm_responses_r: RObj,
    irt_model_r: RObj,
    estimation_method: str = "BM",
) -> float:
    return catr.thetaEst(it=irt_model_r, x=lm_responses_r, method=estimation_method)[0]


def random_accuracy(
    lm_responses: Responses,
    sample_idxes: Idxes,
) -> float:
    lm_responses = np.array(lm_responses)
    sample_idxes = np.array(sample_idxes)
    return np.mean(lm_responses[sample_idxes])


def random_ability(
    lm_responses: Responses,
    irt_model_r: RObj,
    sample_idxes: Idxes,
    estimation_method: str = "BM",
) -> float:

    # Hide items not in random subset
    lm_responses = pd.Series(lm_responses)
    lm_responses[~lm_responses.index.isin(sample_idxes)] = ro.NA_Real

    lm_responses_r = ro.IntVector(lm_responses)
    return catr.thetaEst(it=irt_model_r, x=lm_responses_r, method=estimation_method)[0]


def fluid_benchmarking(
    lm_responses_r: RObj,
    irt_model_r: RObj,
    start_ability: float,
    n_max: int,
    selection_method: str = "MFI",
    estimation_method: str = "BM",
) -> Tuple[List[float], List[int]]:

    # Define hyperparameters
    start = ro.ListVector({"theta": start_ability, "startSelect": selection_method})
    test = ro.ListVector({"method": estimation_method, "itemSelect": selection_method})
    stop = ro.ListVector({"rule": "length", "thr": n_max})
    final = ro.ListVector({"method": estimation_method})

    # Run fluid benchmarking
    eval_fb = catr.randomCAT(
        itemBank=irt_model_r,
        responses=lm_responses_r,
        start=start,
        test=test,
        stop=stop,
        final=final
    )

    # Extract ability estimates and items
    abilities_fb = list(eval_fb.rx("thetaProv")[0])
    items_fb = [int(i) - 1 for i in list(eval_fb.rx2("testItems"))]  # Convert to 0-based indexing
    return abilities_fb, items_fb


def iterate_evals(
    lm_responses: Responses,
    methods: Sequence[str],
    irt_model: Optional[pd.DataFrame] = None,
    estimation_method_irt: str = "BM",
    samples_dict: Optional[Dict[int, np.ndarray]] = None,
    start_ability_fb: float = 0,
    selection_method_fb: str = "MFI",
    seed: int = 0,
) -> Dict[str, Union[float, List[float], List[int]]]:

    # Move to R
    if any(m in config.IRT_METHODS for m in methods):
        if irt_model is None:
            raise ValueError("irt_model is required for IRT-based methods.")
        irt_model_r = rutils.pandas2r(irt_model.reset_index(drop=True))
        lm_responses_r = ro.IntVector(lm_responses)

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
            lm_responses_r, 
            irt_model_r,
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
                irt_model_r,
                samples_dict[n_samples],
                estimation_method_irt
            )

    # Fluid Benchmarking
    if "fluid_benchmarking" in methods:
        abilities_fb, items_fb = fluid_benchmarking(
            lm_responses_r,
            irt_model_r,
            start_ability_fb,
            max(samples_dict.keys()),
            selection_method_fb,
            estimation_method_irt,
        )
        output["abilities_fb"] = abilities_fb
        output["items_fb"] = items_fb

    return output
