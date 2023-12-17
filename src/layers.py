import numpy as np


class Dense():
    """
    Densely-connected layer class.

    This layer implements the operation: `outputs = inputs * weights + biases`.

    Params:
        n_neurons (int): Number of neurons in the layer.
        n_inputs (int): Length of the data sample.
    """

    def __init__(self, n_neurons: int, n_inputs: int) -> None:
        self.weights = 0.1 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs: list) -> None:
        self.inputs = np.array(inputs)
        self.outputs = self.inputs @ self.weights.T + self.biases

    def info(self) -> None:
        print(f"Inputs:\n{self.inputs}\n")
        print(f"Weights:\n{self.weights}\n")
        print(f"Biases:\n{self.biases}\n")
        print(f"Outputs:\n{self.outputs}\n")


class ReLU():
    """
    Rectified Linear Unit activation layer class.

    It returns element-wise `max(0, x)`.
    """

    def forward(self, inputs: list) -> None:
        self.inputs = np.array(inputs)
        self.outputs = np.maximum(0, self.inputs)

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
        self.outputs = np.exp(self.inputs) / np.sum(np.exp(self.inputs))

    def info(self) -> None:
        print(f"Inputs:\n{self.inputs}\n")
        print(f"Outputs:\n{self.outputs}\n")
