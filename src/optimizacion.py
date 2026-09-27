import numpy as np
from scipy.optimize import minimize

from .modelo import logistic_risk, logistic_gradient


def ajustar_modelo(x, y, theta_inicial=(0.0, 0.0), maxiter=1000):
    """
    Ajusta un modelo de regresión logística minimizando
    el riesgo empírico logístico.

    Parámetros
    ----------
    x : array-like
        Datos de entrada.
    y : array-like
        Etiquetas binarias.
    theta_inicial : tuple
        Punto inicial para la optimización.
    maxiter : int
        Número máximo de iteraciones.

    Retorna
    -------
    scipy.optimize.OptimizeResult
        Resultado de scipy.optimize.minimize.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    funcion = lambda theta: logistic_risk(theta, x, y)

    gradiente = lambda theta: logistic_gradient(theta, x, y)

    resultado = minimize(
        funcion,
        np.asarray(theta_inicial, dtype=float),
        jac=gradiente,
        method="BFGS",
        options={
            "maxiter": maxiter
        }
    )

    return resultado


def norma_theta(theta):
    """
    Calcula la norma euclídea de theta.
    """
    return np.linalg.norm(theta)