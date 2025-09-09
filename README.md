<div align="center">
  <h1>Fluid Language Model Benchmarking</h1>
</div>

<p align="center">
 <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/badge/Apache-2.0-D22128?logo=apache&logoColor=white">
 </a>
 <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
 </a>
  <a href="https://huggingface.co/datasets/allenai/fluid-benchmarking">
    <img src="https://img.shields.io/badge/Hugging_Face-Data-yellow?logo=huggingface&logoColor=white">
  </a>
 <a href="https://arxiv.org/abs/TBD">
    <img src="https://img.shields.io/badge/ArXiv-TBD-B31B1B?logo=arxiv&logoColor=white">
  </a>
 <a href="https://allenai.org/blog/TBD">
    <img src="https://img.shields.io/badge/Ai2-Blog-F0529C?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAEIUlEQVR4nO3dsZEcRRTH4V5KDmGQAAngkYBkEQE2OeAqB%2BHjIWKQfUmASQB4i8OVrlTcn9u9mXmvp7%2Bvat29Vs/%2B5s2spLnL9XodwH/7qnoB0JlAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBDIfj6OMa4HvNiRQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEFyuV//vfydHbezloJ%2BzJBMEAoFAIJB9uG49iTfVCyjydof3/H6M8dMO7/tS344xvtnx/R/GGH/s%2BP4trXiT/naM8Vv1Ijb0eJN%2BxIFc7gsBl1gQCAQCgUAgkLn9Xr2AsxPI3Pb4No4nBAKBQCAQyLz%2Brl7ACgQyr6%2BrF7ACgUAgkDkt908%2BqggEAoHMx/Q4kEDmIo6DCQQCgczD9CggkDmIo4hA%2BhNHIYH0Jo5iqz60oTthNGGC9COORkyQPoTRkAlS7zLmiWO5Z0StOEEein/%2BLDE85zrm/zO82IoPjjurigP559j%2BhPPLGOPjxu95N4Gcx5kOZJsJ1e0Sq/ogtzkw9NAtkGpPAxULvsUKrv%2B%2BPHtqYd3uQVot5gvdJ0rnvbtVm702QV7ucaKwEIHcrmsk31Uv4IxcYt2vzWXAEzPtX9Jmb02Q%2B53lw0ggkNfpFkmbM%2B9ZCOT1ukXChgSyjZ%2BrF/DE%2B%2BoFnImb9O10uryZeR/HaLSXJsh2On0o23zAZicQCASyLVPkZARybiJ5JYFsr9MUGUMkryKQNYjkTgJZh0juIJC1XMYY76oXMRN/Ubif7mfsznvdZu9MkHXN9MC6MgJBKIGnmvDoy0g6X4IdRiA8x1QZ/QLZ86A4I3Kzle5BZj8jfhifn6xS9VpOtwmyt8uY70DPtt5TWS2QWYiiiZUusR79WL2A/yGORlacIH9VL%2BAZwmhoxQnSkTiaEkg9cTQmkFriaE4gdcQxAYHUEMckBHI8cUxEIBAI5Fimx2QEcpxfqxfA7QRynB%2BqF8DtBAKBQI7h3mNSAoFAIPvz65knJpD9fapeAPcTCAQCgUAgEAgEghUDeaheAPPo9usPzujoDZ79AXmtrDhBzkwcGxPIeYhjBwI5B3HsRCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIHhTvYAFeCTPxEwQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCwT9pWpVuCH9MegAAAABJRU5ErkJggg%3D%3D&logoWidth=20&labelColor=555555">
  </a>
</p>


## 🛠️ Setup

```sh
git clone https://github.com/allenai/fluid-benchmarking
cd fluid-benchmarking
```

With `conda`:

```sh
conda create -n fluid-benchmarking python=3.10
conda activate fluid-benchmarking
```

With `virtualenv`:

```sh
python -m virtualenv -p python3.10 fluid-benchmarking
source fluid-benchmarking/bin/activate
```
 
Base install:

```sh
python -m pip install -e .
```

If you also need IRT training support, run:

```sh
python -m pip install -e ".[irt]"
```



## 🚀 Usage

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

We provide code in [`irt/fit_irt_model.py`](https://github.com/allenai/fluid-benchmarking/blob/main/irt/fit_irt_model.py) to fit 2PL IRT models with [py-irt](https://github.com/nd-ball/py-irt). The language model evaluation results used as input should be in the form of a JSONL file where each line looks like
`{"subject_id": "lm_1", "responses": {"item_1": 1, "item_2": 0, ...}}` (see the py-irt documentation for details). The output is a CSV file containing the IRT model parameters and can be directly used for Fluid Benchmarking.


### Replicating Experiments From Paper

To replicate the main experiments from the paper, you can use the code in [`scripts/run_experiments.py`](https://github.com/allenai/fluid-benchmarking/blob/main/scripts/run_experiments.py). The script evaluates pretraining checkpoints of [Amber-6.7B](https://huggingface.co/LLM360/Amber), [K2-65B](https://huggingface.co/LLM360/K2), [OLMo1-7B](https://huggingface.co/allenai/OLMo-7B-0724-hf), [OLMo2-7B](https://huggingface.co/allenai/OLMo-2-1124-7B), [Pythia-2.8B](https://huggingface.co/EleutherAI/pythia-2.8b), and [Pythia-6.9B](https://huggingface.co/EleutherAI/pythia-6.9b) on [ARC Challenge](https://huggingface.co/datasets/allenai/ai2_arc), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag), [MMLU](https://huggingface.co/datasets/cais/mmlu), [TruthfulQA](https://github.com/sylinrl/TruthfulQA), and [WinoGrande](https://huggingface.co/datasets/allenai/winogrande), using the methods Random, Random IRT, and Fluid Benchmarking (all with different sample sizes) from the paper, as well as full-benchmark accuracy and IRT ability estimation. The output is stored as JSONL and pickle files in [`results/`](https://github.com/allenai/fluid-benchmarking/tree/main/results). We include the files from the runs analyzed in the paper ([`results/experiments.jsonl`](https://github.com/allenai/fluid-benchmarking/blob/main/results/experiments.jsonl), [`results/experiments.pkl`](https://github.com/allenai/fluid-benchmarking/blob/main/results/experiments.pkl)).

### Replicating Analyses From Paper

We provide code to replicate the main analyses from the paper in `notebooks/analysis.ipynb`.


## 🗂️ Data

The IRT models for the six benchmarks and the language model evaluation results live in a public [Hugging Face dataset](https://huggingface.co/datasets/allenai/fluid-benchmarking). Convenience loaders are provided in [`fluid_benchmarking/datasets.py`](https://github.com/allenai/fluid-benchmarking/blob/main/fluid_benchmarking/datasets.py). For example, IRT models can be loaded as follows:

```python
from fluid_benchmarking import datasets

benchmark = "mmlu"
irt_model = datasets.load_irt_model(
    repo_id="allenai/fluid-benchmarking",
    filename=f"data/irt_models/{benchmark}.csv",
)
```

Also in the dataset:

- Accuracy scores and IRT ability estimates for the 102 language models from the
  [Open LLM Leaderboard](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/archive) used in the paper:
  [`data/open_llm_leaderboard_results.json`](https://huggingface.co/datasets/allenai/fluid-benchmarking/blob/main/data/open_llm_leaderboard_results.json)
- A mapping from item IDs to question text and answer options:
  [`data/id_to_item_map.json`](https://huggingface.co/datasets/allenai/fluid-benchmarking/blob/main/data/id_to_item_map.json)


## 📚 Citation

```
@inproceedings{hofmann2025fluid,
  title={Fluid Language Model Benchmarking},
  author={Valentin Hofmann and David Heineman and Ian Magnusson and Kyle Lo and Jesse Dodge and Maarten Sap and Pang Wei Koh and Chun Wang and Hannaneh Hajishirzi and Noah A. Smith},
  booktitle={Second Conference on Language Modeling},
  year={2025}
}
```