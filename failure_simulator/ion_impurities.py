import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, weibull_min
from scipy.optimize import curve_fit

# Constants
A = 1e6  # Pre-exponential factor
Ea = 60000  # Activation energy (J/mol)
R = 8.314  # Gas constant (J/mol·K)
T_25C = 298.15  # 25°C in Kelvin
T_70C = 343.15  # 70°C in Kelvin
T_85C = 358.15  # 85°C in Kelvin

# Function to calculate the reaction rate constant k(T)
def k(T):
    return A * np.exp(-Ea / (R * T))

# Function to calculate the effect of impurities
def f(c, T):
    w = np.exp(-1 / (T_85C - T_25C) * (T - T_25C) + 2)
    return 10 * np.log(1 + w * c)

# Function to calculate the overall reaction rate
def r(T, c):
    return k(T) * (1 + f(c, T))

# Calculate scaling factor at 70°C and 0.01 M concentration
scaling_factor = 1.0 / r(T_70C, 0.01)

# Temperatures (in Celsius)
temperatures_celsius = np.linspace(25, 85, 100)
# Convert to Kelvin
temperatures_kelvin = temperatures_celsius + 273.15

# Impurity concentrations for the main plot
concentrations = [0, 0.01, 0.02, 0.03, 0.04]

# Plotting the results
plt.figure(figsize=(6, 4))

for c in concentrations:
    rates = r(temperatures_kelvin, c) * scaling_factor
    plt.plot(temperatures_celsius, rates, label=f'Impurity concentration: {c:.4g} M')

# Highlight the rate at 70°C
plt.axvline(x=70, color='red', linestyle='--', label='70°C')

# Set logarithmic scale for y-axis
plt.yscale('log')

# Adding labels and title
plt.xlabel('Temperature (°C)')
plt.ylabel('Reaction Rate (log scale)')
plt.title('Effect of Temperature and Ion Impurities on Corrosion Rate')
plt.legend()
plt.grid(True)
plt.show()

# Generate normal distribution for molar concentrations centered at 0.01 M with std 0.005
mean_concentration = 0.01
std_concentration = 0.01
concentration_samples = np.random.normal(mean_concentration, std_concentration, 1000)

# Calculate reaction rates at 70°C for the concentration samples
rates_70C = r(T_70C, concentration_samples) * scaling_factor

# Fit Weibull distribution to the rate data
params = weibull_min.fit(rates_70C, floc=0)

# Plotting the reaction rate distribution with Weibull fit
plt.figure(figsize=(6, 4))
plt.hist(rates_70C, bins=30, density=True, alpha=0.6, color='g', label='Rate Distribution at 70°C')

""" # Define the Weibull probability density function using the fitted parameters
x = np.linspace(min(rates_70C), max(rates_70C), 100)
pdf_weibull = weibull_min.pdf(x, *params)

plt.plot(x, pdf_weibull, 'r-', lw=2, label=f'Weibull Fit: k={params[0]:.2f}, λ={params[2]:.2f}')
plt.xlabel('Reaction Rate (relative to 1.0 at 0.01 M)')
plt.ylabel('Probability Density')
plt.title('Reaction Rate Distribution at 70°C with Weibull Fit')
plt.legend()
plt.grid(True)
plt.show() """

# Fit Gaussian distribution to the rate data
mu, sigma = norm.fit(rates_70C)

# Define the Gaussian probability density function using the fitted parameters
x = np.linspace(min(rates_70C), max(rates_70C), 100)
pdf_gaussian = norm.pdf(x, mu, sigma)

plt.plot(x, pdf_gaussian, 'r-', lw=2, label=f'Gaussian Fit: μ={mu:.2f}, σ={sigma:.2f}')
plt.xlabel('Reaction Rate (relative to 1.0 at 0.01 M)')
plt.ylabel('Probability Density')
plt.title('Reaction Rate Distribution at 70°C with Gaussian Fit')
plt.legend()
plt.grid(True)
plt.show()


# Print reaction rates at 25°C, 70°C, and 85°C for different concentrations
print("Reaction rates at 25°C, 70°C, and 85°C for different impurity concentrations:")
for c in concentrations:
    rate_25C = r(T_25C, c) * scaling_factor
    rate_70C = r(T_70C, c) * scaling_factor
    rate_85C = r(T_85C, c) * scaling_factor
    print(f"Concentration: {c:.4f} M - Rate at 25°C: {rate_25C:.2e} times, Rate at 70°C: {rate_70C:.2e} times, Rate at 85°C: {rate_85C:.2e} times")
