import torch

from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation
from flwr.common import Context

import client_app
import server_app
from parser_config import get_cfg, parse_args
from task import train, test, tune
import ray


DEVICE = torch.device("cuda")
NUM_CLIENTS = 4


def main() -> None:

    args, unknown = parse_args()
    cfg = get_cfg()

    def make_client_fn(cfg):
        def client_fn_with_cfg(context:Context):
            return client_app.client_fn(context, cfg)
        return client_fn_with_cfg

    def make_server_fn(cfg):
        def server_fn_with_cfg(context:Context):
            return server_app.server_fn(context, cfg)
        return server_fn_with_cfg

    if args.mode == "train":
        train(cfg)
    elif args.mode == "test":
        test(cfg, args.visualize)
    elif args.mode == "tune":
        tune(cfg)
    elif args.mode == "FL":

        backend_config = {"client_resources": None}
        print(DEVICE.type)
        if DEVICE.type == "cuda":
            backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1}}
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            _temp_dir="/tmp/ray",
            object_store_memory=1_000_000_000,
            num_gpus=1,
        )
        server = ServerApp(server_fn=make_server_fn(cfg))
        client = ClientApp(client_fn=make_client_fn(cfg))
        

        run_simulation(
            server_app=server,
            client_app=client,
            num_supernodes=NUM_CLIENTS,
            backend_config=backend_config,
        )



if __name__ == "__main__":
    main()
