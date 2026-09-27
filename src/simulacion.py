import numpy as np

from .modelo import sigmoid


def generar_datos(n, rng):
    """
    Genera datos según el modelo:

        X ~ U[-2, 2]

        P(Y=1|X=x) = sigmoid(-1 + 2x)

        Y|X ~ Bernoulli(sigmoid(-1 + 2X))
    """

    x = rng.uniform(-2, 2, size=n)

    probabilidades = sigmoid(-1 + 2 * x)

    y = rng.binomial(
        n=1,
        p=probabilidades
    )

    return x, y


def generar_datos_modelo(n, seed=2026):
    """
    Genera datos utilizando una semilla determinada.
    """
    rng = np.random.default_rng(seed)

    return generar_datos(n, rng)