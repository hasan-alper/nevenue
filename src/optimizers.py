import numpy as np


class SGD():
    """
    Stochastic Gradient Descent optimizer class.

    Params:
        learning_rate (float): Learning rate. Defaults to 1.0.
        decay (float): Decay rate. Defaults to 0.0.
        momentum (float): Momentum rate. Defaults to 0.0.
    """

    def __init__(self, learning_rate: float = 1.0, decay: float = 0.0, momentum: float = 0.0) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self) -> None:
        self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer: object) -> None:
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases

            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self) -> None:
        self.iterations += 1
