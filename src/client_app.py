from utils import load_config
from yacs.config import CfgNode
from task import federatedTrain, evaluate_model
import flwr as fl
from flwr.client import NumPyClient
import torch
import numpy as np
from typing import List
import argparse
from models.build_model import Build_Model
from data.unified_loader import unified_loader
import os
import sys

# --- BEGIN sys.path MODIFICATION ---
# This assumes your environment.py is located at:
# /home/emiliano/Documents/Flowchain-training/flowchain-implementation/src/data/TP/environment.py

# Get the path to the directory that client_app.py is in (e.g., .../src/)
base_src_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the directory DIRECTLY CONTAINING 'environment.py'
# This allows 'import environment' to find it if environment.py is in this directory.
path_to_environment_module_dir = os.path.join(
    base_src_dir,
    "data",
    "TP",  # This is .../src/
)

# Add this path to the beginning of sys.path if it's not already there
if path_to_environment_module_dir not in sys.path:
    sys.path.insert(0, path_to_environment_module_dir)
    print(f"INFO: Added to sys.path for dill: {
          path_to_environment_module_dir}")
# --- END sys.path MODIFICATION ---


class FlowerClient(NumPyClient):
    def __init__(self, local_epochs, cfg: CfgNode):
        self.net = Build_Model(cfg)
        self.local_epochs = local_epochs
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.cfg = cfg
        self.partition_id = cfg.get("partition_id", None)
        self.trainloader = unified_loader(
            cfg=cfg,
            rand=True,
            split="train",
            batch_size=128,
        )
        self.valloader = unified_loader(
            cfg=cfg,
            rand=False,
            split="test",  # CAN CHANGE
            batch_size=128,
        )

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        # Grab the model’s existing state dict (an OrderedDict of all keys)
        state_dict = self.net.state_dict()
        keys = list(state_dict.keys())

        # Sanity‐check: are we even the same length?
        if len(parameters) > len(keys):
            raise ValueError(
                f"Too many parameter arrays ({len(parameters)}) "
                f"for this model ({len(keys)} keys)."
            )

        # Overwrite in order, leave any extra keys untouched
        for key, array in zip(keys, parameters):
            state_dict[key] = torch.tensor(array)

        # Now load back in (strict=True is safe: keys match exactly)
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        server_round = config.get("current_round")
        results = federatedTrain(
            cfg=self.cfg,
            model=self.net,
            train_loader=self.trainloader,
        )
        metrics = evaluate_model(
            self.cfg, self.net, self.valloader, self.cfg.TEST.VISUALIZE, server_round
        )
        ade = metrics.get("ade", 0.0)
        fde = metrics.get("fde", 0.0)
        metrics["loss"] = results
        metrics["score"] = ade
        print(
            f"[FedRep Fit Client] Round {server_round} - ADE: {ade:.4f}, FDE: {
                fde:.4f
            }, "
            f"Score: {metrics['score']:.4f}, Inference time: {metrics['inference_time']:.6f}s")

        return (
            self.get_parameters(config={}),
            len(self.trainloader),
            metrics,
        )

    def evaluate(self, parameters, config):
        server_round = config.get("current_round")
        self.set_parameters(parameters)
        metrics = evaluate_model(
            self.cfg, self.net, self.valloader, self.cfg.TEST.VISUALIZE, server_round
        )
        ade = metrics.get("ade", 0.0)
        fde = metrics.get("fde", 0.0)
        metrics["score"] = ade
        print(
            f"[FedRep Evaluate Client] Round {server_round} - ADE: {ade:.4f}, FDE: {
                fde:.4f
            }, "
            f"Score: {metrics['score']:.4f}, Inference time: {metrics['inference_time']:.6f}s")

        return (
            float(metrics["score"]),
            len(self.valloader),
            metrics,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning Client for FlowChain"
    )
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to config YAML"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Whether to visualize predictions"
    )
    parser.add_argument("--gpu", type=str, default="0",
                        help="CUDA GPU device id")
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Mode: train/test/tune (default: train)",
    )
    parser.add_argument("--partition_id", required=True,
                        help="This client's ID")
    args = parser.parse_args()

    cfg = load_config(args)
    cfg.defrost()
    cfg.TEST.VISUALIZE = args.visualize
    cfg.DATA.NUM_WORKERS = 0
    cfg.freeze()

    client = FlowerClient(local_epochs=cfg.SOLVER.ITER, cfg=cfg)

    fl.client.start_client(server_address="0.0.0.0:9090",
                           client=client.to_client())


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
