from yacs.config import CfgNode
from task import Net
from task import federatedTrain, federatedTest, load_dataset 
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from collections import OrderedDict


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



class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, cfg: CfgNode, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.cfg = cfg
        self.partition_id = partition_id

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit() called — loading parameters, about to train...")
        set_parameters(self.net, parameters)
        print(f"[Client {self.partition_id}]  train loader size: {len(self.trainloader)}  |  batch size: {self.trainloader.batch_size}")
        results = federatedTrain(
            cfg = self.cfg,
            model = self.net,
            data_loader = self.trainloader,
            val_data_loader= self.valloader,
            device= self.device,
        )
        print(f"[Client {self.partition_id}] fit() returning — trained for {self.local_epochs} local epochs")
        return get_parameters(self.net), len(self.trainloader.dataset), {"result": results}


    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        result_info = federatedTest(
            cfg = self.cfg,
            model = self.net,
            visualize= False,
        )
        return float(result_info["ade"]), len(self.valloader), {"Result_info": result_info["score"]}


def client_fn(context: Context, cfg: CfgNode):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    local_epochs = cfg.SOLVER.get("LOCAL_EPOCHS", cfg.SOLVER.ITER)
    net = Net(cfg).to(device)
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, valloader, _ = load_dataset(cfg, client_id=partition_id, num_clients=num_partitions)
    return FlowerClient(net, trainloader, valloader, local_epochs=local_epochs, cfg=cfg, partition_id=partition_id).to_client()
