import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, norm

# Parameters for the Weibull distribution
k = 2.0  # Shape parameter
lam = 208  # Scale parameter

# Create a range of x values
x = np.linspace(0, 500, 400)

# Calculate Weibull PDF
weibull_pdf = weibull_min.pdf(x, k, scale=lam)

# Parameters for the Gaussian distribution, calculated to fit the original Weibull distribution
mu_original = np.average(x, weights=weibull_pdf)  # Weighted mean of Weibull PDF
sigma_original = np.sqrt(np.average((x - mu_original)**2, weights=weibull_pdf))  # Weighted standard deviation

# Calculate Gaussian PDF with adjusted parameters
gaussian_pdf_original = norm.pdf(x, mu_original, sigma_original)

# Plotting both curves for comparison
plt.figure(figsize=(10, 6))
plt.plot(x, weibull_pdf, label='Original Weibull PDF', color='red')
plt.plot(x, gaussian_pdf_original, label='Adjusted Gaussian PDF', color='blue')
plt.title('Comparison of Original Weibull and Gaussian PDFs')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()

# Display the Gaussian parameters on the plot
plt.text(400, 0.002, f'Gaussian Mean = {mu_original:.2f}\nGaussian Std Dev = {sigma_original:.2f}', bbox=dict(facecolor='white', alpha=0.5))

plt.show()
