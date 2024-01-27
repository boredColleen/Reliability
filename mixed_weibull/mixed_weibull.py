import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Define the negative log-likelihood for the Weibull distribution
def neg_log_likelihood(params, times, events, include_censored):
    shape, scale = params
    uncensored_likelihood = np.where(events == 1, 
                                     np.log(shape) - np.log(scale) + (shape - 1) * (np.log(times) - np.log(scale)) - (times / scale) ** shape, 
                                     0)
    censored_likelihood = np.where(events == 0, 
                                   - (times / scale) ** shape, 
                                   0)
    return -np.sum(uncensored_likelihood + censored_likelihood * include_censored)

# Define the likelihood function for theta
def neg_log_likelihood_theta(theta, time_grid, sf_values_full, sf_values_observed, times, events):
    sf_values_mixed = (1 - theta) * sf_values_full + theta * sf_values_observed
    pdf_values_mixed = -np.gradient(sf_values_mixed, time_grid)
    pdf_interp = interp1d(time_grid, pdf_values_mixed, kind='linear', fill_value='extrapolate')
    pdf_values_at_observed = pdf_interp(times[events == 1])
    return -np.sum(np.log(pdf_values_at_observed))

# Sample data
times = np.array([800, 1000, 1300, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400])
events = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
initial_guess = [1.0, 1000.0]

# Minimize negative log-likelihood for observed and full data
result_observed = minimize(neg_log_likelihood, initial_guess, args=(times, events, False), bounds=[(0.001, None), (0.001, None)])
result_full = minimize(neg_log_likelihood, initial_guess, args=(times, events, True), bounds=[(0.001, None), (0.001, None)])
shape_est_observed, scale_est_observed = result_observed.x
shape_est_full, scale_est_full = result_full.x

# Calculate failure rate and survival functions
fr = np.mean(events)
time_grid = np.linspace(0, max(times) + 200, 1000)
sf_values_observed = fr * np.exp(-(time_grid / scale_est_observed) ** shape_est_observed) + (1 - fr)
sf_values_full = np.exp(-(time_grid / scale_est_full) ** shape_est_full)

# Minimize negative log-likelihood for theta
initial_guess_theta = [0.5]
result_theta = minimize(neg_log_likelihood_theta, initial_guess_theta, args=(time_grid, sf_values_full, sf_values_observed, times, events), bounds=[(0, 1)])
theta_est = 1#result_theta.x[0]
print(f"Estimated Theta: {theta_est}")

# Plot survival function
sf_values_mixed = (1 - theta_est) * sf_values_full + theta_est * sf_values_observed
plt.figure(figsize=(5, 4))
plt.plot(time_grid, sf_values_mixed, color='green', label='Weibull SF')
plt.xlabel('Time')
plt.ylabel('SF')
plt.title('Weibull SF')
plt.legend()
plt.xlim(-50, 1450)
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.show()
