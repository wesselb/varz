import lab as B

__all__ = ["ADAM"]


class ADAM:
    """Bare-bones implementation of ADAM.

    Args:
        rate (float, optional): Learning rate. Defaults to `1e-3`.
        beta1 (float, optional): Exponential decay for mean. Defaults to `0.9`.
        beta2 (float, optional): Exponential decay for second moment. Defaults to
            `0.999`.
        epsilon (float, optional): Small value to prevent division by zero.
            Defaults to `1e-8`.
        local_rates (bool, optional): Use local learning rates. Set to `False` to
            use one global learning rate. Defaults to `True`.
    """

    def __init__(
        self, rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, local_rates=False
    ):
        self.rate = rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.local_rates = local_rates

        self.m = None
        self.v = None
        self.i = 0

    def step(self, x, grad):
        """Perform a gradient step.

        Args:
            x (tensor): Current input value. This value will be updated in-place.
            grad (tensor): Current gradient.

        Returns:
            tensor: `x` after updating `x` in-place.
        """
        if self.m is None or self.v is None:
            self.m = B.zeros(x)
            self.v = B.zeros(x)

        # Update estimates of moments.
        self.m *= self.beta1
        self.m += (1 - self.beta1) * grad
        self.v *= self.beta2
        self.v += (1 - self.beta2) * grad ** 2

        # Correct for bias of initialisation.
        m_corr = self.m / (1 - self.beta1 ** (self.i + 1))
        v_corr = self.v / (1 - self.beta2 ** (self.i + 1))

        # Perform update.
        if self.local_rates:
            denom = B.sqrt(B.mean(v_corr)) + self.epsilon
        else:
            denom = B.sqrt(v_corr) + self.epsilon
        x -= self.rate * m_corr / denom

        # Increase iteration number.
        self.i += 1

        return x
