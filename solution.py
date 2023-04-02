import numpy as np
from scipy.stats import lognorm
from scipy.optimize import minimize


chat_id = 356550601

def solution(x: np.array) -> float:
    def log_likelihood(alpha, sigma_sq, x):
        return np.sum(np.log(251 + lognorm(sigma_sq, scale=np.exp(alpha)).pdf(x)))

    result = minimize(lambda params: -log_likelihood(*params, x), [0, 1])
    return result.x[0]
