from flwr.server.strategy import FedAvg
import flwr as fl
import argparse

NUM_CLIENTS = 4


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Server (FedAvg)")
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

    # Initialize FedAvg with round tracking

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=NUM_CLIENTS,
    )

    # Start Flower server
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
