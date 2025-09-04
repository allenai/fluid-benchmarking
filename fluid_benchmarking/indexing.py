import pandas as pd


def id2benchmark(
    item_id: str,
) -> str:
    benchmark = "_".join(item_id.split("_")[:-1])
    return "mmlu" if benchmark.startswith("mmlu") else benchmark


def filter_benchmark(
    lm_eval_results: pd.DataFrame, 
    benchmark: str,
) -> pd.DataFrame:
    mask = lm_eval_results.index.map(lambda x: id2benchmark(x) == benchmark)
    return lm_eval_results[mask]
