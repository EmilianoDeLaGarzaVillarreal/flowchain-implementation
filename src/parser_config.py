import argparse
from utils import load_config
from yacs.config import CfgNode


def parse_args():
    parser = argparse.ArgumentParser(
        description="pytorch training & testing code for task-agnostic time-series prediction")
    parser.add_argument("--config_file", type=str, default='',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "tune", "FL"], default="FL")
    parser.add_argument(
        "--visualize", action="store_true", help="flag for whether visualize the results in mode:test")

    return parser.parse_known_args()


def get_cfg() -> CfgNode:
    args, unknown = parse_args()
    return load_config(args, unknown)
