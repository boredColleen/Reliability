import numpy as np
from scipy.optimize import minimize
from scipy.stats import weibull_min
import matplotlib.pyplot as plt

# Given x values
x = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])

# Given y values for three different curves
y1 = np.array([100, 100, 100, 100, 98, 90, 40, 20, 10, 0]) / 100
y2 = np.array([100, 90, 85, 70, 68, 28, 20, 14, 8, 2]) / 100
y3 = np.array([100, 40, 25, 16, 6, 2, 1, 0, 0, 0]) / 100

# Objective function to minimize
def objective(params):
    c1, c2, c3, scale = params
    y_fit_1 = 1 - weibull_min.cdf(x, c1, scale=scale)
    y_fit_2 = 1 - weibull_min.cdf(x, c2, scale=scale)
    y_fit_3 = 1 - weibull_min.cdf(x, c3, scale=scale)
    # Sum of squared errors
    error = np.sum((y1 - y_fit_1) ** 2) + np.sum((y2 - y_fit_2) ** 2) + np.sum((y3 - y_fit_3) ** 2)
    return error

# Initial guess for the parameters
initial_guess = [1, 1, 1, 100]

# Perform the optimization
result = minimize(objective, initial_guess)
c1_opt, c2_opt, c3_opt, scale_opt = result.x

# Generate a dense x array for plotting
x_dense = np.linspace(min(x), max(x), 300)

# Plot the original data and the fitted curves
plt.figure(figsize=(12, 8))
plt.scatter(x, y1, color='red', label='y1 Original')
plt.plot(x_dense, 1 - weibull_min.cdf(x_dense, c1_opt, scale=scale_opt), 'r--', label='y1 Fitted')
plt.scatter(x, y2, color='green', label='y2 Original')
plt.plot(x_dense, 1 - weibull_min.cdf(x_dense, c2_opt, scale=scale_opt), 'g--', label='y2 Fitted')
plt.scatter(x, y3, color='blue', label='y3 Original')
plt.plot(x_dense, 1 - weibull_min.cdf(x_dense, c3_opt, scale=scale_opt), 'b--', label='y3 Fitted')

plt.title('Fitted Weibull Curves with Shared Scale Factor')
plt.xlabel('X')
plt.ylabel('Survival Probability')
plt.legend()
plt.show()
