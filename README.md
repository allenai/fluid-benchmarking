<div align="center">
  <h1>Fluid Language Model Benchmarking</h1>
</div>

<p align="center">
 <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
 </a>
 <a href="https://www.r-project.org/">
    <img src="https://img.shields.io/badge/R-4.1-green?logo=r&logoColor=white">
 </a>
 <a href="https://arxiv.org/abs/TBD">
    <img src="https://img.shields.io/badge/ArXiv-TBD-B31B1B?logo=arxiv&logoColor=white">
  </a>
 <a href="https://allenai.org/blog/TBD">
    <img src="https://img.shields.io/badge/Ai2-Blog-F0529C?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAJE0lEQVR4nOzde4xcZfnA8WcuO3vfnd3ftr%2BWVhDszQZ6WdkmphqlBDWYKA3BYhATEBKNBgIBrZhgkIiJ0SjBRKOmf0mCmghSTRpIBWnLJdBLim2V3hYK7ba73e3sbmdm58ycY%2BZoNy5ln3a355z3XL6fZBISkjMPYb8z58zM%2B55sS2%2BfIwA%2BUNr0AECYEQigIBBAQSCAgkAABYEACgIBFAQCKAgEUBAIoCAQQEEggIJAAAWBAAoCARQEAigIBFAQCKAgEEBBIICCQAAFgQAKAgEUBAIoCARQEAigIBBAQSCAgkAABYEACgIBFAQCKAgEUBAIoMiaHiCuPpRpk50LbvH1OZ4aPyj3DG%2F39TmSjncQQEEggIJAAAWBAAoCARQEAigIBFAQCKAgEEBBIICCQAAFgQAKAgEUBAIoCARQEAigIBBAQSCAgkAABYEACgLxSXOK%2FTDigEB8srih0%2FQI8ACB%2BOT2tiWmR4AHCMQHD3SsknXNC02PAQ8k7kQ5JSLd6SbPj9uYSsvyhm75StsSubHlCs%2BPfyH5VE4yKf9e7yzHllGn4tvxwypxgdTjOLDwy6bH8NzmeTfK0oYu346%2Fo3xC1p%2Fa4tvxw4pTrAgrJPAVPWgEEmGnaiXTI8QegUTYQatgeoTYI5AI21cZNj1C7BFIRJ2sFuVYbdz0GLFHIBG1pzJkeoREIJCIemXipOkREoFAIuq50jHTIyQCgUTQK%2BUBOVTlE6wgEEgEPTTymukREoNAIub50jHZZ%2FHxblAIJGJ%2BObrP9AiJQiARsmnsgGyfOGF6jEQhkIjYXxmWjSOvmh4jcQgkAoq2JfcP7zA9RiIRSMhZji23D22VXXxzbgSBhFg9jm%2Befkm2lbnuMCVxKwqjouzU5LbB54nDMAIJoYPWGbnn9HbZWRk0PUriEUjIPDn%2Bljww%2FLLUxDE9CggkPHaUT8jPR%2FfKS%2BXjoUwj5%2BOOKWFGIIbtnhiUR868IS9PDJgeRbWsoVsykkrcOxuBBGzMrriLnbaXB2Rr6V150zodiT%2B59nSDbOzslR8WdpoeJVCJC8RybPd0xm%2B2ODJuW%2B6nUcdrRTlsFWR3ZUgOWCPuv4uieztXuEE%2FW%2Bw3PUpgUi29fdH8v4Upts2%2FydeN486ZcGry9Nkj8kzxqPRXx9wXHC8VnaqctsueHvNSJO4dJK7Kdi2Q52lMZeTWtsXuww9%2FLh6Vu4de9OXYsxGqQJ6e%2B7nAnqv%2BSjVqV%2BRkrSRvV8dkvzUseyvDUnKqgc3gpYoTTCBJE6pA1jbNN%2Fr8Vcd213r%2Femx%2F6D9Ver9he8L0CLEUqkBMy6bS7s7s9cchq%2BB%2BaffU2UOhOieezkCtaHqEWErmtz8XYVFDp3y%2Fq09evexmua01%2FDfDqV8ww3sEcgGd6Zz87P%2FWyu%2FnfEaWZvOmx5nWAWvE9AixRCAX6brmBfLC%2FC%2FK55uDvznOxdg9wXoRPxDIDNSvUX7V8ym5v2Ol%2B7OLMBlxJuTF0numx4gdApmhxlRGNuZ73UjC5jdj%2B02PEDsEMkv3da6UeztWmB5jiq3l96TfGjU9RqwQyCzVT7e%2Bl%2F%2BYfKH5w6ZHmWSLI384e9j0GLFCIJfo0a41Mi%2FTYnqMSU%2BefUuKdjR%2FDRBGBHKJ5mdbZVPPdZIOyUX7iVpRHkvYT9L9RCAeuLZxrmxo%2FYjpMSb9duyAbCsdNz1GLBCIR77RfrXpESbVr0UePvO66TFigUA8sizXJRtaFpkeY9I%2Ba1juGnzB8%2FUaSUMgHvpuvjc01yJ1z5b65aeFPabHiDQC8dBl2Vb5eOP%2Fmx5jisdH98qPC7vdn%2FJj5gjEY9c3LzQ9whQ1ceQnhT3yo8Iu06NEEoF4LGzvIOc8Mfqm3DH4N3m3yr3VZ4JAPLYi1xO6HzKe89fS23LDwOZAdnWJCwLxWEMqLb25OabHmNZpuyzrT22Rrw%2F93d20DjoC8cHqxh7TI1zQn4pH5LMn%2FyJfHdwq77AacVqsSffBFdl20yNctC2ld9xHX26urGteIH2Nc2V1rkfa0znTo4UCgfhgXrrZ9Agz9nrllPs4Z2lDXq7JdcvlmXaZk2mW1nQwfyq7QrYyMlSB%2BHFO3JLOuq%2FoTang%2FlOzMdgJ%2FV%2FWGfeRdKEKpH5O7IfGVEYeyffJne0f9eX4iK9QBeKXCafm3kJ5R3lAftHzSWkO8N3EKwsyrbIq1yNLGvJyZfY%2Fpz3d6cbAnn9PZUi%2Bk8DbUEfvL%2BUSbC71y9rxeZF5J8lISr7Uukjubl8uV%2Be6jc5SjOiWrJcqUYHUPVM8GvpAlmbzckf7MlnfcpV0ZYJ7l8D5EhfI4RBvalB%2Fx3g0v0bu6lhuehT8V%2BICCevNa25oWuiub7%2BqodP0KPgfiQskjB7sWCUP5lebHgMfgEAMykpaHu9eK7e0hWclIqYiEEPqcfxuzvWyLmTrRzBV9L%2Fyjagnuj9BHBFAIAbc17FSbm4LzzZBmB6BBOzTTZfJtzu5II8KAgnYD%2FJrJJMK54pDnI9AAvRw57Xu%2FlmIDgIJyMJMm3yr8xrTY2CGCCQgd7YvMz0CZoFAAnJr62LTI2AWCCQAN7VcKT2ZJtNjYBYIJABfawv3z%2BsxPQLxWVe60b1%2FCKKJQHy2KtfD9x4RRiA%2BW8L6jkgjEJ9dnonOJnI4H4H4bA6fXkUagfiMLTyjLXELpizH9n37%2F%2F3WyOQ%2FN6Uyvj7X%2B52sFeVIdVRsx9u19%2F%2BoDHt6vKhIXCCjTsXd%2Fj9uxuyKPDTymvzx7OHQbkwRRYkLJI6KtiUbTj0nb1S434fXuAaJgccKu4jDJwQScQV7QjaN%2FdP0GLFFIBF3rDouVeEWz34hkIizuP%2B5rwgEUBAIoCAQQEEggIJAAAWBAAoCARQEAigIBFAQCKAgEEBBIICCQAAFgQAKAgEUBAIoCARQEAigIBBAwb5YPttRHpAhu%2Bzb8futUd%2BODZFUS28f2%2FAB0%2BAUC1AQCKAgEEBBIICCQAAFgQAKAgEUBAIoCARQEAigIBBAQSCAgkAABYEACgIBFAQCKAgEUBAIoCAQQEEggIJAAAWBAAoCARQEAigIBFAQCKAgEEBBIICCQAAFgQAKAgEUBAIoCARQEAgwPeffAQAA%2F%2F%2BVRx%2FfL810FQAAAABJRU5ErkJggg%3D%3D&logoColor=white">
  </a>
  <a href="https://huggingface.co/datasets/allenai/fluid-benchmarking">
    <img src="https://img.shields.io/badge/Hugging_Face-Data-yellow?logo=huggingface&logoColor=white">
  </a>
 <a href="TBD">
    <img src="https://img.shields.io/badge/License-TBD-orange?logo=opensourceinitiative&logoColor=white">
 </a>
</p>


## 🛠️ Setup

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

We provide code in [`irt/fit_irt_model.py`](https://github.com/allenai/fluid-benchmarking/blob/main/irt/fit_irt_model.py) to fit 2PL IRT models with py-irt. The language model evaluation results used as input should be in the form of a JSONL file where each line looks like
`{"subject_id": "lm_1", "responses": {"item_1": 1, "item_2": 0, ...}}` (see the [py-irt documentation](https://github.com/nd-ball/py-irt) for details). The output is a CSV file containing the IRT model parameters and can be directly used for Fluid Benchmarking.


### Replicating Experiments From Paper

To replicate the main experiments from the paper, you can use the code in [`scripts/run_experiments.py`](https://github.com/allenai/fluid-benchmarking/blob/main/scripts/run_experiments.py). The script evaluates [Amber-6.7B](https://huggingface.co/LLM360/Amber), [K2-65B](https://huggingface.co/LLM360/K2), [OLMo1-7B](https://huggingface.co/allenai/OLMo-7B-0724-hf), [OLMo2-7B](https://huggingface.co/allenai/OLMo-2-1124-7B), [Pythia-2.8B](https://huggingface.co/EleutherAI/pythia-2.8b), and [Pythia-6.9B](https://huggingface.co/EleutherAI/pythia-6.9b) on [ARC Challenge](https://huggingface.co/datasets/allenai/ai2_arc), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag), [MMLU](https://huggingface.co/datasets/cais/mmlu), [TruthfulQA](https://github.com/sylinrl/TruthfulQA), and [WinoGrande](https://huggingface.co/datasets/allenai/winogrande), using the methods Random, Random IRT, and Fluid Benchmarking (all with different sample sizes) from the paper, as well as full-benchmark accuracy and ability estimation. The output is stored as JSONL and pickle files in [`results/`](https://github.com/allenai/fluid-benchmarking/tree/main/results). We include the files from the runs analyzed in the paper ([`results/experiments.jsonl`](https://github.com/allenai/fluid-benchmarking/blob/main/results/experiments.jsonl), [`results/experiments.pkl`](https://github.com/allenai/fluid-benchmarking/blob/main/results/experiments.pkl)).

### Replicating Analyses From Paper

We provide code to replicate the main analyses from the paper in `notebooks/analysis.ipynb`.


## 🗂️ Data

## 📚 Citation

```
@inproceedings{hofmann2025fluid,
  title={Fluid Language Model Benchmarking},
  author={Valentin Hofmann and David Heineman and Ian Magnusson and Kyle Lo and Jesse Dodge and Maarten Sap and Pang Wei Koh and Chun Wang and Hannaneh Hajishirzi and Noah A. Smith},
  booktitle={Second Conference on Language Modeling},
  year={2025}
}
```