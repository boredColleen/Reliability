import numpy as np
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Given x values
x = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])

# Given y values for three different curves
y1 = np.array([100, 100, 100, 100, 98, 90, 40, 20, 10, 0]) / 100
y2 = np.array([100, 90, 85, 70, 68, 28, 20, 14, 8, 2]) / 100
y3 = np.array([100, 40, 25, 16, 6, 2, 1, 0, 0, 0]) / 100

# Define a function to fit the inverted Weibull CDF for survival probabilities
def inverted_weibull_cdf(x, c, scale):
    return 1 - weibull_min.cdf(x, c, scale=scale)

# Example weights for y1, modify as needed
weights = np.array([4, 2, 10, 1, 1, 1, 1, 1, 1, 1])
weights_inv = 1 / weights  # Inverting weights for curve_fit

# Generate a dense x array for a smoother curve
x_dense = np.linspace(min(x), max(x), 300)

# Initialize figure
plt.figure(figsize=(12, 8))

# Modified function to fit data, plot, and annotate for a given y dataset with weights
def fit_plot_annotate_with_weights(y, label, color, index, weights):
    params, _ = curve_fit(inverted_weibull_cdf, x, y, p0=[1, 100], sigma=weights, absolute_sigma=True)
    y_fitted = inverted_weibull_cdf(x_dense, *params)
    plt.plot(x_dense, y_fitted, label=f'Fitted {label}', color=color)
    plt.scatter(x, y, color=color, s=10)
    
    # Calculate the y position for annotation based on the index
    y_pos = 0.95 - index * 0.05
    plt.annotate(f'{label}: c={params[0]:.2f}, scale={params[1]:.2f}',
                 xy=(0.05, y_pos), xycoords='axes fraction', color=color)

# Fit, plot, and annotate for y1, y2, y3 with weights
fit_plot_annotate_with_weights(y1, 'y1', 'red', 1, weights_inv)
fit_plot_annotate_with_weights(y2, 'y2', 'green', 2, weights_inv)
fit_plot_annotate_with_weights(y3, 'y3', 'blue', 3, weights_inv)

# Finalizing plot
plt.title('Weighted Weibull CDF Survival Curve Fitting for y1, y2, and y3')
plt.xlabel('X')
plt.ylabel('Survival Probability')
plt.legend()

plt.show()
