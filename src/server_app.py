from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from yacs.config import CfgNode
from task import Net
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
from parser_config import get_cfg
from task import load_dataset, federatedTest

NUM_PARTITIONS = 10


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    # Grab the model’s existing state dict (an OrderedDict of all keys)
    state_dict = net.state_dict()
    keys       = list(state_dict.keys())

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
    net.load_state_dict(state_dict, strict=True)


DEVICE = torch.device("cuda")

def server_fn(context: Context, cfg: CfgNode) -> ServerAppComponents:
    # The `evaluate` function will be called by Flower after every round
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = Net(cfg).to(DEVICE)
        _, _, testloader = load_dataset(cfg, -1, NUM_PARTITIONS)
        set_parameters(net, parameters)  # Update model with the latest parameters
        score, ade, fde, num = list(federatedTest(cfg, net, False).values())
        print("Score:", score, "ade:", float(ade), "fde:", float(fde))
        return float(score), {"ade": float(ade), "fde": float(fde)}

   # Create the FedAvg strategy
    params = get_parameters(Net(cfg))
    strategy = FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.3,
        min_fit_clients=5,
        min_evaluate_clients=3,
        min_available_clients=NUM_PARTITIONS,
        initial_parameters=ndarrays_to_parameters(params),
        evaluate_fn=evaluate,  # Pass the evaluation function
    )
    # Configure the server for 3 rounds of training
    config = ServerConfig(num_rounds=5)
    return ServerAppComponents(strategy=strategy, config=config)
