import numpy as np


class Dense():
    """
    Densely-connected layer class.

    This layer implements the operation: `outputs = inputs * weights + biases`.

    Params:
        n_neurons (int): Number of neurons in the layer.
        n_inputs (int): Length of the data sample.
        l1_w (float): L1 regularization parameter for weights.
        l1_b (float): L1 regularization parameter for biases.
        l2_w (float): L2 regularization parameter for weights.
        l2_b (float): L2 regularization parameter for biases.
    """

    def __init__(self, n_neurons: int, n_inputs: int, l1_w: float = 0, l1_b: float = 0, l2_w: float = 0, l2_b: float = 0) -> None:
        self.weights = 0.1 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros(n_neurons)
        self.l1_w = l1_w
        self.l1_b = l1_b
        self.l2_w = l2_w
        self.l2_b = l2_b

    def forward(self, inputs: list) -> None:
        self.inputs = np.array(inputs)
        self.outputs = self.inputs @ self.weights.T + self.biases

    def backward(self, dvalues: list) -> None:
        self.dweights = dvalues.T @ self.inputs  # Gradient of weights
        self.dbiases = np.sum(dvalues, axis=0)  # Gradient of biases

        if self.l1_w:
            self.dweights += self.l1_w * np.sign(self.weights)  # L1 regularization gradient of weights
        if self.l1_b:
            self.dbiases += 2 * self.l1_b * np.sign(self.biases)  # L1 regularization gradient of biases
        if self.l2_w:
            self.dweights += self.l2_w * self.weights  # L2 regularization gradient of weights
        if self.l2_b:
            self.dbiases += 2 * self.l2_b * self.biases  # L2 regularization gradient of biases

        self.dinputs = dvalues @ self.weights  # Gradient of inputs

    def info(self) -> None:
        print(f"Inputs:\n{self.inputs}\n")
        print(f"Weights:\n{self.weights}\n")
        print(f"Biases:\n{self.biases}\n")
        print(f"Outputs:\n{self.outputs}\n")


class Dropout():
    """
    Dropout layer class.

    Dropout is a regularization technique that randomly sets a fraction of the input units to 0 at each training step.

    Params:
        rate (float): Fraction of the input units to drop.
    """

    def __init__(self, rate: float) -> None:
        self.rate = 1 - rate

    def forward(self, inputs: list) -> None:
        self.inputs = np.array(inputs)
        self.binary_mask = np.random.binomial(1, self.rate, size=self.inputs.shape) / self.rate
        self.outputs = self.inputs * self.binary_mask

    def backward(self, dvalues: list) -> None:
        self.dinputs = dvalues * self.binary_mask  # Gradient of inputs


class ReLU():
    """
    Rectified Linear Unit activation layer class.

    It returns element-wise `max(0, x)`.
    """

    def forward(self, inputs: list) -> None:
        self.inputs = np.array(inputs)
        self.outputs = np.maximum(0, self.inputs)

    def backward(self, dvalues: list) -> None:
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0  # Gradient of inputs

    def info(self) -> None:
        print(f"Inputs:\n{self.inputs}\n")
        print(f"Outputs:\n{self.outputs}\n")


class Softmax():
    """
    Softmax activation layer class.

    Softmax converts a vector of values to a probability distribution. It returns `exp(x) / sum(exp(x))`.
    """

    def forward(self, inputs: list) -> None:
        self.inputs = np.array(inputs)
        exp_values = np.exp(self.inputs - np.max(self.inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities

    def backward(self, dvalues: list) -> None:
        self.dinputs = np.empty_like(dvalues)

        for i, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalues)

    def info(self) -> None:
        print(f"Inputs:\n{self.inputs}\n")
        print(f"Outputs:\n{self.outputs}\n")


class Sigmoid():
    """
    Sigmoid activation layer class.

    It returns element-wise `1 / (1 + exp(-x))`.
    """

    def forward(self, inputs: list) -> None:
        self.inputs = np.array(inputs)
        self.outputs = 1 / (1 + np.exp(-self.inputs))

    def backward(self, dvalues: list) -> None:
        self.dinputs = dvalues * (1 - self.outputs) * self.outputs  # Gradient of inputs

    def info(self) -> None:
        print(f"Inputs:\n{self.inputs}\n")
        print(f"Outputs:\n{self.outputs}\n")
