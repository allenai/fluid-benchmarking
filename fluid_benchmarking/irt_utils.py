import numpy as np


def sigmoid_stable(
    z: np.ndarray,
) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def fisher_information(
    theta: float, 
    a: np.ndarray, 
    b: np.ndarray, 
    D: float = 1.0,
) -> np.ndarray:
    z = D * a * (theta - b)
    P = sigmoid_stable(z)
    return (D ** 2) * (a ** 2) * (P * (1.0 - P))
