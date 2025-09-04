import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyro
import torch
from py_irt.config import IrtConfig
from py_irt.dataset import Dataset
from py_irt.training import IrtModelTrainer

from two_param_logistic import TwoParamLogistic


def main():

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1000,
    )
    args = parser.parse_args()

    # Make reproducible
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)

    # Process input and output paths
    input_path = Path(args.input_path).expanduser()
    print(input_path)
    params_dir = Path(__file__).resolve().with_name("params")
    params_dir.mkdir(parents=True, exist_ok=True)
    out_csv = params_dir / f"{input_path.stem}.csv"
    
    # Load input data
    data = Dataset.from_jsonlines(str(input_path))
    config = IrtConfig(
        model_type=TwoParamLogistic,
        priors="hierarchical",
    )

    # Fit IRT model
    trainer = IrtModelTrainer(
        config=config, 
        data_path=None, 
        dataset=data
    )
    trainer.train(epochs=args.epochs, device=args.device)

    # Extract IRT parameters
    discriminations = [np.exp(i) for i in trainer.best_params["disc"]]  # Since parameter in log space
    difficulties = list(trainer.best_params["diff"])
    item_ids = trainer.best_params["item_ids"].values()
    irt_model = pd.DataFrame({"a": discriminations, "b": difficulties}, index=item_ids)
    irt_model.to_csv(out_csv)


if __name__ == "__main__":
    main()
