import numpy as np

def prelu_forward(x: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """
    Implements the forward pass of PReLU.
    Args:
        x: Input array of any shape
        alpha: Slope parameter for negative values (default: 0.25)
    Returns:
        np.ndarray: PReLU activation output, same shape as x
    """

    output = np.where(
        x > 0,
        x,
        x * alpha
    )

    return output


def prelu_backward(
    x: np.ndarray,
    alpha: float,
    grad_output: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Implements the backward pass of PReLU.
    """

    grad_x = grad_output * np.where(
        x > 0,
        1,
        alpha
    )

    grad_alpha = np.sum(
        grad_output[x <= 0] * x[x <= 0]
    )

    return grad_x, grad_alpha