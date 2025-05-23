import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    Parameters, Scalar, FitIns, FitRes, EvaluateIns, EvaluateRes,
    MetricsAggregationFn, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
)

class FedCMB(fl.server.strategy.Strategy):
    """Federated Learning strategy with Combined Metric-Based Aggregation (FedCMB)."""

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        params = self.initial_parameters
        self.initial_parameters = None
        return params

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, config)

        sample_size, min_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(sample_size, min_clients)

        return [(client, fit_ins) for client in clients]

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        if self.fraction_evaluate == 0.0:
            return []

        config = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, min_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(sample_size, min_clients)

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Extract parameters and compute weights
        parameters_list = [parameters_to_ndarrays(res.parameters) for _, res in results]
        weights, total_weight = [], 0

        print(f"\n===== Round {server_round} Aggregation Info =====")

        for idx, (client, res) in enumerate(results):
            client_id = getattr(client, "cid", f"Client-{idx}")
            combined_metric = self.compute_combined_metric(res)

            # Compute inverse weight
            epsilon = 1e-10
            weight = 1.0 / (combined_metric if combined_metric > epsilon else epsilon)

            weights.append(weight)
            total_weight += weight

            print(f"Client {client_id} | ADE: {res.metrics.get('ade', 0):.4f} | "
                  f"FDE: {res.metrics.get('fde', 0):.4f} | "
                  f"Combined Metric: {combined_metric:.4f} | Weight: {weight:.4f}")

        # Normalize weights
        normalized_weights = [w / total_weight for w in weights]

        print("\nNormalized Weights per Client:")
        for idx, (client, _) in enumerate(results):
            client_id = getattr(client, "cid", f"Client-{idx}")
            print(f"Client {client_id}: {normalized_weights[idx]:.4f}")
        print("=====================================\n")

        # Weighted aggregation
        aggregated_params = []
        for idx in range(len(parameters_list[0])):
            param_slices = [params[idx] for params in parameters_list]
            weighted_param = np.average(param_slices, axis=0, weights=normalized_weights)
            aggregated_params.append(weighted_param)

        aggregated_parameters = ndarrays_to_parameters(aggregated_params)
        return aggregated_parameters, {"round": server_round}

    def compute_combined_metric(self, fit_res: Union[FitRes, Tuple]) -> float:
        """Compute harmonic mean of ADE and FDE."""
        try:
            metrics = fit_res[1].metrics if isinstance(fit_res, tuple) else fit_res.metrics
            print(f"[DEBUG] Received metrics from client: {metrics}")
            fde = float(metrics.get("fde", 0) or 0)
            ade = float(metrics.get("ade", 0) or 0)

            if fde + ade == 0:
                print(f"[WARNING] Both FDE and ADE are zero or missing: fde={fde}, ade={ade}")
                return 1.0  # Avoid division by zero

            return 2 * (fde * ade) / (fde + ade)
        except Exception as e:
            print(f"[ERROR] Failed to compute combined metric: {e}")
            return 1.0

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        total_loss = sum(res.loss for _, res in results if res)
        num_clients = len(results)
        avg_loss = total_loss / num_clients if num_clients else None
        return avg_loss, {"avg_loss": avg_loss} if avg_loss is not None else (None, {})

    def evaluate(self, server_round: int, parameters: Parameters):
        """Evaluate global model parameters using an evaluation function."""
        if not self.evaluate_fn:
            return None
        params_ndarrays = parameters_to_ndarrays(parameters)
        result = self.evaluate_fn(server_round, params_ndarrays, {})
        return result

