import flwr as fl
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy.fedavg import FedAvg


class FedRepCustom(FedAvg):
    """FedRep logic: only aggregates representation parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Each result is (client, FitRes)
        weights = []
        num_examples = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weights.append(ndarrays)
            num_examples.append(fit_res.num_examples)

        # Weighted average of parameters (representation only)
        total_weight = sum(num_examples)
        weighted_weights = [
            [layer * num_examples[i] for layer in weights[i]]
            for i in range(len(weights))
        ]

        aggregated = [
            sum(layer_updates) / total_weight for layer_updates in zip(*weighted_weights)
        ]

        return ndarrays_to_parameters(aggregated), {}
