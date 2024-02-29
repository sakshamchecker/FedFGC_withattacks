
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import (
	EvaluateRes,
	Scalar,
)
from flwr.server.client_proxy import ClientProxy




def weighted_avg(results: List[Tuple[int, float, Optional[float]]]) -> float:
	"""Aggregate evaluation results obtained from multiple clients."""
	num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
	weighted_values = [num_examples * value for num_examples, value in results]
	
	return sum(weighted_values) / num_total_evaluation_examples


class FedAvgWithAccuracyMetric(fl.server.strategy.FedAvg):
	def aggregate_evaluate(self,
						   rnd: int,
						   results: List[Tuple[ClientProxy, EvaluateRes]],
						   failures: List[BaseException],
						   ) -> Tuple[Optional[float], Dict[str, Scalar]]:
		"""Aggregate evaluation losses using weighted average."""
		
		if not results:
			return None, {}
		# Do not aggregate if there are failures and failures are not accepted
		if not self.accept_failures and failures:
			return None, {}
		loss_aggregated = weighted_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])
		accuracy_aggregated = weighted_avg([(evaluate_res.num_examples, evaluate_res.metrics['accuracy']) for _, evaluate_res in results])
		return loss_aggregated, {"accuracy": accuracy_aggregated}

class FedProxWithAccuracyMetric(fl.server.strategy.FedProx):
	def aggregate_evaluate(self,
						   rnd: int,
						   results: List[Tuple[ClientProxy, EvaluateRes]],
						   failures: List[BaseException],
						   ) -> Tuple[Optional[float], Dict[str, Scalar]]:
		"""Aggregate evaluation losses using weighted average."""
		
		if not results:
			return None, {}
		# Do not aggregate if there are failures and failures are not accepted
		if not self.accept_failures and failures:
			return None, {}
		loss_aggregated = weighted_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])
		accuracy_aggregated = weighted_avg([(evaluate_res.num_examples, evaluate_res.metrics['accuracy']) for _, evaluate_res in results])
		return loss_aggregated, {"accuracy": accuracy_aggregated}


class FedOptAdamStrategy(fl.server.strategy.FedAvg):
	def aggregate(self, reports):
		# Extract model parameters from reports
		parameters = [report.parameters for report in reports]

		# Initialize aggregated model parameters
		aggregated_parameters = [torch.zeros_like(param) for param in parameters[0]]

		# Perform FedOptAdam aggregation
		for param_group in zip(*parameters):
			# Compute the mean of the parameters in the group
			mean_param = torch.mean(torch.stack(param_group), dim=0)

			# Update the aggregated parameters with FedOptAdam logic
			for i, param in enumerate(param_group):
				alpha = 0.1  # Adjust the learning rate if needed
				aggregated_parameters[i] += alpha * (param - mean_param)

		# Compute the final aggregated model parameters
		aggregated_parameters = [param / len(reports) for param in aggregated_parameters]

		return aggregated_parameters
	def aggregate_evaluate(self,
						   rnd: int,
						   results: List[Tuple[ClientProxy, EvaluateRes]],
						   failures: List[BaseException],
						   ) -> Tuple[Optional[float], Dict[str, Scalar]]:
		"""Aggregate evaluation losses using weighted average."""
		
		if not results:
			return None, {}
		# Do not aggregate if there are failures and failures are not accepted
		if not self.accept_failures and failures:
			return None, {}
		loss_aggregated = weighted_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])
		accuracy_aggregated = weighted_avg([(evaluate_res.num_examples, evaluate_res.metrics['accuracy']) for _, evaluate_res in results])
		return loss_aggregated, {"accuracy": accuracy_aggregated}
def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config

def evaluate_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config