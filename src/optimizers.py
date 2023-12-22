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


class AdaGrad():
    """
    AdaGrad optimizer class.

    Params:
        learning_rate (float): Learning rate. Defaults to 1.0.
        decay (float): Decay rate. Defaults to 0.0.
        epsilon (float): Epsilon value. Defaults to 1e-7.
    """

    def __init__(self, learning_rate: float = 1.0, decay: float = 0.0, epsilon: float = 1e-7) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self) -> None:
        self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer: object) -> None:
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self) -> None:
        self.iterations += 1


class RMSProp:
    """
    RMSProp optimizer class.

    Params:
        learning_rate (float): Learning rate. Defaults to 0.001.
        decay (float): Decay rate. Defaults to 0.0.
        epsilon (float): Epsilon value. Defaults to 1e-7.
        rho (float): Rho value. Defaults to 0.9.
    """

    def __init__(self, learning_rate: float = 0.001, decay: float = 0.0, epsilon: float = 1e-7, rho: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self) -> None:
        self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer: object) -> None:
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self) -> None:
        self.iterations += 1


class Adam():
    """
    Adam optimizer class.

    Params:
        learning_rate (float): Learning rate. Defaults to 0.001.
        decay (float): Decay rate. Defaults to 0.0.
        epsilon (float): Epsilon value. Defaults to 1e-7.
        beta_1 (float): Beta 1 value. Defaults to 0.9.
        beta_2 (float): Beta 2 value. Defaults to 0.999.
    """

    def __init__(self, learning_rate: float = 0.001, decay: float = 0.0, epsilon: float = 1e-7, beta_1: float = 0.9, beta_2: float = 0.999) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self) -> None:
        self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer: object) -> None:
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self) -> None:
        self.iterations += 1
