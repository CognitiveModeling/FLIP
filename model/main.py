import argparse
import sys
import torch as th
import numpy as np
import os

from utils.configuration import Configuration
from model.scripts import training  # Keep training import as is

import json

def update_paths(data, scratch_path):
    lockfile = "/tmp/data_copy.lock"

    # Define datasets
    datasets = ['train', 'val', 'test']

    # Iterate over each dataset
    for dataset in datasets:
        if dataset not in data['data']:
            continue

        for item in data['data'][dataset]:

            new_paths = []
            for source_path in item['paths']:
                # Replace /mnt/lustre/butz/mtraub38 with scratch_path in the source path
                updated_path = source_path.replace('/scratch', scratch_path)
                print(f"Updating: {source_path} -> {updated_path}")
                new_paths.append(updated_path)

            item['paths'] = new_paths

    return data


CFG_PATH = "cfg.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg", default=CFG_PATH)
    parser.add_argument("-num-gpus", "--num-gpus", default=1, type=int)
    parser.add_argument("-n", "--n", default=-1, type=int)
    parser.add_argument("-load", "--load", default="", type=str)
    parser.add_argument("-load-model-only", "--load-model-only", action="store_true")
    parser.add_argument("-scratch", "--scratch", type=str, default="")
    parser.add_argument("-device", "--device", default=0, type=int)
    parser.add_argument("-single-gpu", "--single-gpu", action="store_true")
    parser.add_argument("-seed", "--seed", default=1234, type=int)
    parser.add_argument("-validate", "--validate", action="store_true")
    parser.add_argument("-float32-matmul-precision", "--float32-matmul-precision", default="highest", type=str)

    args = parser.parse_args(sys.argv[1:])

    th.set_float32_matmul_precision(args.float32_matmul_precision)

    cfg = Configuration(args.cfg)
    if args.scratch != "":
        cfg = update_paths(cfg, args.scratch)

    cfg.single_gpu = args.single_gpu

    cfg.seed = args.seed
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)

    cfg.validate = args.validate

    if args.device >= 0:
        cfg.device = args.device
        cfg.model_path = f"{cfg.model_path}.device{cfg.device}"

    if args.n >= 0:
        cfg.model_path = f"{cfg.model_path}.run{args.n}"

    num_gpus = th.cuda.device_count()

    if cfg.device >= num_gpus:
        cfg.device = num_gpus - 1

    if args.num_gpus > 0:
        num_gpus = args.num_gpus

    # Handle training and evaluation modes
    training.train(cfg, args.load if args.load != "" else None, args.load_model_only)
