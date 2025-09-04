# Hugging Face dataset repo ID
HF_REPO_ID = "allenai/fluid-benchmarking"

# Path templates
LM_EVAL_RESULTS_PATH = "data/lm_eval_results/{}.csv"
IRT_MODELS_PATH = "data/irt_models/{}.csv"

# Supported LMs, benchmarks, and methods
LMS = [
    "amber-7b",
    "k2-65b",
    "olmo1-7b",
    "olmo2-7b",
    "pythia-7b",
    "pythia-3b"
]

BENCHMARKS = [
    "arc_challenge", 
    "gsm8k", 
    "hellaswag",
    "truthfulqa_mc2",
    "winogrande",
    "mmlu"
]

METHODS = [
    "full_accuracy",
    "full_ability",
    "random_accuracy",
    "random_ability",
    "fluid_benchmarking"
]

IRT_METHODS = [
    "full_ability", 
    "random_ability", 
    "fluid_benchmarking"
]

# Default parameters for IRT-based methods and Fluid Benchmarking
ESTIMATION_METHOD_IRT = "BM"
SELECTION_METHOD_FB = "MFI"

# Default evaluation sample sizes
N_SAMPLES_LIST = (
    list(range(1, 10)) + 
    list(range(10, 100, 10)) + 
    list(range(100, 600, 100))
)
