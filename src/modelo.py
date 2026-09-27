import numpy as np
from scipy.special import expit


def sigmoid(t):
    """
    Función logística:
    
        sigma(t) = 1 / (1 + exp(-t))
    
    Se utiliza scipy.special.expit para obtener
    una evaluación numéricamente estable.
    """
    return expit(t)


def predict_proba(theta, x):
    """
    Calcula p_theta(x) para un conjunto de datos x.

    Parámetros
    ----------
    theta : array-like
        Vector [theta0, theta1].
    x : array-like
        Valores de la variable explicativa.

    Retorna
    -------
    numpy.ndarray
        Probabilidades estimadas.
    """
    theta = np.asarray(theta, dtype=float)
    x = np.asarray(x, dtype=float)

    theta0, theta1 = theta

    return sigmoid(theta0 + theta1 * x)


def logistic_risk(theta, x, y):
    """
    Calcula el riesgo empírico logístico:

        R_S(theta) =
        -(1/n) sum[
            y_i log(p_i) +
            (1-y_i) log(1-p_i)
        ]

    Se utiliza una expresión numéricamente estable.
    """
    theta = np.asarray(theta, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    theta0, theta1 = theta

    z = theta0 + theta1 * x

    # Forma estable de la pérdida logística:
    # log(1 + exp(z)) - y*z
    loss = np.logaddexp(0, z) - y * z

    return np.mean(loss)


def logistic_gradient(theta, x, y):
    """
    Gradiente del riesgo logístico respecto de theta0 y theta1.
    """
    theta = np.asarray(theta, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    p = predict_proba(theta, x)

    error = p - y

    gradient_theta0 = np.mean(error)
    gradient_theta1 = np.mean(error * x)

    return np.array([
        gradient_theta0,
        gradient_theta1
    ])


def log_likelihood(theta, x, y):
    """
    Calcula la log-verosimilitud:

        log L(theta) =
        sum[
            y_i log(p_i) +
            (1-y_i) log(1-p_i)
        ]
    """
    p = predict_proba(theta, x)

    # Evitamos log(0) por redondeo numérico.
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1 - eps)

    return np.sum(
        y * np.log(p) +
        (1 - y) * np.log(1 - p)
    )


def likelihood(theta, x, y):
    """
    Calcula directamente la verosimilitud:

        L(theta) = producto[
            p_i^y_i (1-p_i)^(1-y_i)
        ]

    Esta función puede producir 0.0 por underflow
    cuando el producto es extremadamente pequeño.
    """
    p = predict_proba(theta, x)

    likelihood_terms = np.where(
        y == 1,
        p,
        1 - p
    )

    return np.prod(likelihood_terms)