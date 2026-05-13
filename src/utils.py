# Helpers for loading config, setting seeds, and saving models

import os
import random
import numpy as np
import yaml
import joblib


def load_config(path="config.yaml"):
    """Reads the YAML config file and returns it as a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    """Makes sure we get the same results every time we run the code."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def save_model(model, path):
    """Saves the trained model to a file so we can load it later."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """Loads a previously saved model from disk."""
    return joblib.load(path)


def ensure_dirs(config):
    """Creates output folders if they dont already exist."""
    for key in ["model_dir", "plot_dir", "metrics_dir"]:
        os.makedirs(config["paths"][key], exist_ok=True)
