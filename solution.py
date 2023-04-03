import numpy as np
from scipy.optimize import minimize

chat_id = 356550601

def solution(x: np.array) -> float:
    def negative_log_likelihood(params, x):
        alpha, sigma_sq = params
        return -np.sum(251 + np.log(1 / (x * np.sqrt(2 * np.pi * sigma_sq))) - ((np.log(x) - alpha) ** 2) / (2 * sigma_sq))

    res = minimize(negative_log_likelihood, (0, 1), args=x, method='Nelder-Mead')
    return res.x[0]
