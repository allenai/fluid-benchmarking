import argparse
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tqdm

from fluid_benchmarking import config, datasets, evaluation, indexing


def run_experiments(
    benchmarks: Iterable[str],
    lms: Iterable[str],
    seed: int = 0,
) -> pd.DataFrame:

    # Load evaluation results
    lm_eval_results = {}
    for lm in lms:
        lm_eval_results[lm] = datasets.load_lm_eval_results(
            config.HF_REPO_ID,
            config.LM_EVAL_RESULTS_PATH.format(lm),
        )

    # Load Open LLM Leaderboard results for estimating start abilities
    open_llm_leaderboard_results = datasets.load_open_llm_leaderboard_results()

    random.seed(seed)

    rows = []
    for benchmark in tqdm.tqdm(list(benchmarks), desc="Benchmarks"):

        # Load IRT model
        irt_model_benchmark = datasets.load_irt_model(
            config.HF_REPO_ID,
            config.IRT_MODELS_PATH.format(benchmark),
        )

        # Sanity checks
        assert irt_model_benchmark.columns[0] == "a"
        assert irt_model_benchmark.columns[1] == "b"

        # Determine random subset of benchmark items
        samples_dict = {}
        for n_samples in config.N_SAMPLES_LIST:
            samples_dict[n_samples] = np.array(
                random.sample(range(len(irt_model_benchmark)), n_samples)
            )

        # Define start ability for fluid benchmarking (mean over LMs)
        start_ability = float(
            np.mean(list(open_llm_leaderboard_results[benchmark]["ability"].values()))
        )

        for lm in tqdm.tqdm(
            list(lms), 
            desc=f"{benchmark} • LMs", 
            leave=False
        ):
            lm_eval_results_benchmark = indexing.filter_benchmark(
                lm_eval_results[lm],
                benchmark,
            )

            # Sanity check
            assert (lm_eval_results_benchmark.index == irt_model_benchmark.index).all()

            for checkpoint in tqdm.tqdm(
                list(lm_eval_results_benchmark.columns),
                desc=f"{lm} • checkpoints",
                leave=False,
            ):
                lm_responses = np.array(lm_eval_results_benchmark[checkpoint])
                row = evaluation.iterate_evals(
                    lm_responses=lm_responses,
                    methods=config.METHODS,
                    irt_model=irt_model_benchmark,
                    estimation_method_irt=config.ESTIMATION_METHOD_IRT,
                    samples_dict=samples_dict,
                    start_ability_fb=start_ability,
                )
                row["benchmark"] = benchmark
                row["lm"] = lm
                row["checkpoint"] = checkpoint
                rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=config.BENCHMARKS,
        help="Benchmarks to evaluate.",
    )
    parser.add_argument(
        "--lms",
        nargs="+",
        default=config.LMS,
        help="LMs to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subset sampling.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/experiments.jsonl"),
        help="Output file.",
    )
    args = parser.parse_args()

    df = run_experiments(
        benchmarks=args.benchmarks,
        lms=args.lms,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(args.out, orient="records", lines=True)
    df.to_pickle(args.out.with_suffix(".pkl"))
    print(f"Wrote {len(df):,} rows to {args.out} and {args.out.with_suffix('.pkl')}.")


if __name__ == "__main__":
    main()
