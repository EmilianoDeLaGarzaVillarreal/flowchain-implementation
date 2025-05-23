from flwr.server.strategy import FedAvg
import flwr as fl
import argparse

NUM_CLIENTS = 1


def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning Server (FedAvg)")
    parser.add_argument(
        "--num_rounds", type=int, default=5, help="Number of federated training rounds"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Server host IP (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=9090, help="Server port (default: 9090)"
    )

    args = parser.parse_args()

    def evaluate_config(server_round: int) -> dict:
        config = {
            "current_round": server_round,
        }

        return config

    # Initialize FedAvg with round tracking

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_evaluate_config_fn=evaluate_config,
    )

    # Start Flower server
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
