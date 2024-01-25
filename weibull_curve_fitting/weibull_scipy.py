import numpy as np
from scipy.optimize import minimize

# Sample data
times = np.array([800, 1000, 1200])  # Your data
events = np.array([1, 1, 0])  # 1 if event observed, 0 if censored (assuming the last one is censored)

# Define the negative log-likelihood for the Weibull distribution with censoring
def neg_log_likelihood(params):
    shape, scale = params
    # Likelihood for uncensored data
    uncensored_likelihood = np.where(events == 1,
                                     np.log(shape) - np.log(scale) + (shape - 1) * (np.log(times) - np.log(scale)) - (times / scale) ** shape,
                                     0)
    # Likelihood for censored data
    censored_likelihood = np.where(events == 0,
                                   - (times / scale) ** shape,
                                   0)
    # Total likelihood is the sum of uncensored and censored likelihoods
    total_likelihood = uncensored_likelihood + censored_likelihood
    # Take the negative because we will minimize this function
    return -np.sum(total_likelihood)

# Initial guesses for shape and scale
initial_guess = [1.0, 1000.0]

# Minimize the negative log-likelihood
result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B', bounds=[(0.001, None), (0.001, None)])

# Extract the estimated shape and scale parameters
shape_est, scale_est = result.x

print(f"Estimated Shape: {shape_est}")
print(f"Estimated Scale: {scale_est}")
