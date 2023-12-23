import numpy as np


class Loss():
    """
    Base class for loss function classes.
    """

    def regularization_loss(self, layer) -> float:
        """
        Calculates the regularization loss from all the layer's weights and biases.
        """
        regularization_loss = 0

        if layer.l1_w:
            regularization_loss += layer.l1_w * np.sum(np.abs(layer.weights))

        if layer.l1_b:
            regularization_loss += layer.l1_b * np.sum(np.abs(layer.biases))

        if layer.l2_w:
            regularization_loss += layer.l2_w * np.sum(layer.weights * layer.weights)

        if layer.l2_b:
            regularization_loss += layer.l2_b * np.sum(layer.biases * layer.biases)

        return regularization_loss


class CategoricalCrossentropy(Loss):
    """
    Computes the crossentropy loss between the labels and predictions.

    Labels are expected to be in a `one_hot` representation.
    """

    def calculate(self, y_true: list, y_pred: list) -> float:
        y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)

        confidence = np.sum(y_pred * y_true, axis=1)
        loss = -np.log(confidence)
        loss_avg = np.mean(loss)

        return loss_avg

    def backward(self, dvalues: list, y_true: list) -> None:
        n_samples = len(dvalues)

        self.dinputs = -y_true / dvalues  # Gradient of inputs
        self.dinputs = self.dinputs / n_samples  # Average gradient of inputs


class SparseCategoricalCrossentropy(Loss):
    """
    Computes the crossentropy loss between the labels and predictions.

    Labels are expected to be integers.
    """

    def calculate(self, y_true: list, y_pred: list) -> float:
        y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)

        confidence = y_pred[range(len(y_pred)), y_true]
        loss = -np.log(confidence)
        loss_avg = np.mean(loss)

        return loss_avg

    def backward(self, dvalues: list, y_true: list) -> None:
        n_samples = len(dvalues)
        n_labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]  # One-hot encode labels

        self.dinputs = -y_true / dvalues  # Gradient of inputs
        self.dinputs = self.dinputs / n_samples  # Average gradient of inputs
