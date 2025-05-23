from custom_strategy.FedRepCustom import FedRepCustom
import flwr as fl
import argparse
import numpy as np

NUM_CLIENTS = 1


class LoggingFedRep(FedRepCustom):
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
        if not results:
            print(f"[Server - FedRep] Round {server_round} - No evaluation results received.")
            return None, {}

        scores =[]

        for client, eval_result in results:
            metrics = eval_result.metrics or {}
            ade = metrics.get("ade", 0.0)
            fde = metrics.get("fde", 0.0)
            score = metrics.get("score", ade)
            inference_time = metrics.get("inference_time", None)

            log_line = f"Client {client.cid} - ADE: {ade:.4f}, FDE: {fde:.4f}, Score: {score:.4f}"
            if inference_time is not None:
                log_line += f", Inference time: {inference_time:.6f}s"
            print(log_line)

            scores.append(score)

        avg_score = float(np.mean(scores))
        return avg_score, {"avg_Score": avg_score}

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

    strategy = LoggingFedRep(
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
