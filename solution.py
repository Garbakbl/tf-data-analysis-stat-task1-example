import numpy as np
from scipy.optimize import minimize

chat_id = 356550601

def solution(checks):
    def negative_log_likelihood(params, checks):
        alpha, sigma_sq = params
        return -np.sum(251 + np.log(1 / (checks * np.sqrt(2 * np.pi * sigma_sq))) - ((np.log(checks) - alpha) ** 2) / (2 * sigma_sq))

    res = minimize(negative_log_likelihood, (0, 1), args=checks, method='Nelder-Mead')
    return res.x[0]
