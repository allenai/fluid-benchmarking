# Fluid Language Model Benchmarking

## Overview


## Setup

```sh
git clone https://github.com/allenai/fluid-benchmarking
cd fluid-benchmarking
```

The repository depends on Python, R, and rpy2. We recommend using conda to create a clean stack:

```sh
conda env create -f environment.yml
conda activate fluid-benchmarking
```

Install the required packages:

```sh
Rscript -e "install.packages('catR', repos='https://cloud.r-project.org')"
python -m pip install -e .
```

As a quick sanity check, open Python and run:

```pycon
>>> import rpy2.robjects as ro
>>> print("R:", ro.r("R.version.string")[0])
R: R version 4.1.3 (2022-03-10)
>>> print("catR:", ro.r("as.character(packageVersion('catR'))")[0])
catR: 3.17
```

## Usage

The core entry point to Fluid Benchmarking is [`evaluation.fluid_benchmarking()`](https://github.com/allenai/fluid-benchmarking/blob/db30ec8f4b1275978156a473a314cfb73e18beff/fluid_benchmarking/evaluation.py#L57). Given language model evaluation results on a benchmark (`lm_responses`) and corresponding IRT parameters (`irt_model`), Fluid Benchmarking can be conducted as follows:

```python
from fluid_benchmarking import evaluation, rutils

# Convert LM responses and IRT model to rpy2 objects
lm_responses_r = rutils.vector2r(lm_responses)
irt_model_r = rutils.df2r(irt_model)

# Set hyperparameters
start_ability = 0
n_max = 100
selection_method = "MFI"  # Maximum Fisher information
estimation_method = "BM"  # Bayes modal estimation (MAP)

# Run Fluid Benchmarking
abilities_fb, items_fb = evaluation.fluid_benchmarking(
    lm_responses_r=lm_responses_r,
    irt_model_r=irt_model_r,
    start_ability=start_ability,
    n_max=n_max,
    selection_method=selection_method,
    estimation_method=estimation_method,
)
```

`evaluation.fluid_benchmarking()` returns the provisional ability estimates and the administered items from the benchmark, which can be used for further analyses. For a complete, runnable example, see [`notebooks/demo.ipynb`](https://github.com/allenai/fluid-benchmarking/blob/main/notebooks/demo.ipynb).


### Fitting IRT Models

We provide code in [`irt/fit_irt_model.py`](https://github.com/allenai/fluid-benchmarking/blob/main/irt/fit_irt_model.py) to fit 2PL IRT models with py-irt. The language model evaluation results used as input should be in the form of a JSONL file where each line looks like
`{"subject_id": "lm_1", "responses": {"item_1": 1, "item_2": 0, ...}}` (see the [py-irt documentation](https://github.com/nd-ball/py-irt) for details). The output is a CSV file containing the IRT model parameters and can be directly used for Fluid Benchmarking.


### Replicating Experiments From Paper

To replicate the main experiments from the paper, you can use the code in [`scripts/run_experiments.py`](https://github.com/allenai/fluid-benchmarking/blob/main/scripts/run_experiments.py). The script evaluates [Amber-6.7B](https://huggingface.co/LLM360/Amber), [K2-65B](https://huggingface.co/LLM360/K2), [OLMo1-7B](https://huggingface.co/allenai/OLMo-7B-0724-hf), [OLMo2-7B](https://huggingface.co/allenai/OLMo-2-1124-7B), [Pythia-2.8B](https://huggingface.co/EleutherAI/pythia-2.8b), and [Pythia-6.9B](https://huggingface.co/EleutherAI/pythia-6.9b) on [ARC Challenge](https://huggingface.co/datasets/allenai/ai2_arc), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag), [MMLU](https://huggingface.co/datasets/cais/mmlu), [TruthfulQA](https://github.com/sylinrl/TruthfulQA), and [WinoGrande](https://huggingface.co/datasets/allenai/winogrande), using the methods Random, Random IRT, and Fluid Benchmarking (all with different sample sizes) from the paper, as well as full-benchmark accuracy and ability estimation. The output is stored as JSONL and pickle files in [`results/`](https://github.com/allenai/fluid-benchmarking/tree/main/results). We include the files from the runs analyzed in the paper ([`results/experiments.jsonl`](https://github.com/allenai/fluid-benchmarking/blob/main/results/experiments.jsonl), [`results/experiments.pkl`](https://github.com/allenai/fluid-benchmarking/blob/main/results/experiments.pkl)).

### Replicating Analyses From Paper

We provide code to replicate the main analyses from the paper in `notebooks/analysis.ipynb`.


## Data

## Citation

```
@inproceedings{hofmann2025fluid,
  title={Fluid Language Model Benchmarking},
  author={Valentin Hofmann and David Heineman and Ian Magnusson and Kyle Lo and Jesse Dodge and Maarten Sap and Pang Wei Koh and Chun Wang and Hannaneh Hajishirzi and Noah A. Smith},
  booktitle={Second Conference on Language Modeling},
  year={2025}
}
```