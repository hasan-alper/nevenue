import numpy as np


class CategoricalCrossentropy():
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


class SparseCategoricalCrossentropy():
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
