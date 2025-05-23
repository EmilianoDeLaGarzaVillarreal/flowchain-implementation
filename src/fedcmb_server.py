
from custom_strategy.FedCMB import FedCMB
import flwr as fl
import argparse
import numpy as np

NUM_CLIENTS = 1


class LoggingFedCMB(FedCMB): 
    def configure_fit(self, server_round, parameters, client_manager):
        config = {"current_round" : server_round}
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        return [
            (client, fl.common.FitIns(parameters=ins.parameters, config=config))
            for client, ins in fit_ins
        ]


    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {"current_round" : server_round}
        evaluate_ins = super().configure_evaluate(server_round, parameters, client_manager)
        return [
            (client, fl.common.EvaluateIns(parameters=ins.parameters, config=config))
            for client, ins in evaluate_ins
        ]

    def aggregate_evaluate(self, server_round, results, failures):
        metrics = super().aggregate_evaluate(server_round, results, failures)
        print(f"\n[Server - FedCMB] Round {server_round} - Client Evaluation Results:")
        for client, eval_result in results:
            client_metrics = eval_result.metrics
            ade = client_metrics.get("ade", 0.0)
            fde = client_metrics.get("fde", 0.0)
            score = client_metrics.get("score", 0.0)
            inference_time = client_metrics.get("inference_time", None)

            log_line = f"Client {client.cid} - ADE: {ade:.4f}, FDE: {fde:.4f}, Score: {score:.4f}"
            if inference_time is not None:
                log_line += f", Inference time: {inference_time:.6f}s"
            print(log_line)
        return metrics

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

    # Initialize FedAvg with round tracking

    strategy = LoggingFedCMB(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
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
