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