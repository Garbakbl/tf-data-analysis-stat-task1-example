import numpy as np
from scipy.optimize import minimize

chat_id = 356550601

def solution(x: np.array) -> float:
    log_data = np.log(x-251)
    mu = np.mean(log_data)
    return mu
