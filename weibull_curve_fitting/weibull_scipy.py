import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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

# Calculate the PDF and SF for a fine grid of time points for plotting
time_grid = np.linspace(0, max(times) + 200, 1000)
pdf_values = (shape_est / scale_est) * ((time_grid / scale_est) ** (shape_est - 1)) * np.exp(-(time_grid / scale_est) ** shape_est)
sf_values = np.exp(-(time_grid / scale_est) ** shape_est)

# Get PDF and SF values for actual data points for plotting
pdf_data_points = (shape_est / scale_est) * ((times / scale_est) ** (shape_est - 1)) * np.exp(-(times / scale_est) ** shape_est)
sf_data_points = np.exp(-(times / scale_est) ** shape_est)

# Plotting
plt.figure(figsize=(10, 4))

# Plot PDF
plt.subplot(1, 2, 1)
plt.plot(time_grid, pdf_values, color='blue', label='Weibull PDF')
plt.scatter(times[events == 1], pdf_data_points[events == 1], color='red', label='Uncensored Data')  # Uncensored data points on PDF curve
plt.scatter(times[events == 0], pdf_data_points[events == 0], color='green', label='Censored Data')  # Censored data points on PDF curve
plt.xlabel('Time')
plt.ylabel('PDF')
plt.title('Weibull PDF')
plt.legend()

# Plot SF
plt.subplot(1, 2, 2)
plt.plot(time_grid, sf_values, color='green', label='Weibull SF')
plt.scatter(times[events == 1], sf_data_points[events == 1], color='red', label='Uncensored Data')  # Uncensored data points on SF curve
plt.scatter(times[events == 0], sf_data_points[events == 0], color='green', label='Censored Data')  # Censored data points on SF curve
plt.xlabel('Time')
plt.ylabel('SF')
plt.title('Weibull SF')
plt.legend()

plt.tight_layout()
plt.show()
