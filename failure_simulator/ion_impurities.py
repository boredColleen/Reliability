import numpy as np
import matplotlib.pyplot as plt

# Constants
A = 1e6  # Pre-exponential factor (arbitrary units)
Ea = 60000  # Activation energy (J/mol)
R = 8.314  # Gas constant (J/mol·K)
alpha = 100  # Impurity influence factor
beta = 1  # Impurity scaling factor
T0 = 298  # Reference temperature (25°C in Kelvin)
gamma = 0.1  # Steepness of the weight function

# Function to calculate the reaction rate constant k(T)
def k(T):
    return A * np.exp(-Ea / (R * T))

# Function to calculate the effect of impurities
def f(c):
    return np.log(1 + alpha * c)

# Weight function to modulate the effect of temperature and impurities
def w(T):
    return 1 / (1 + np.exp(gamma * (T - T0)))

# Function to calculate the overall reaction rate
def r(T, c):
    return k(T) * (1 + beta * f(c)) * w(T) + k(T) * (1 - w(T))

# Temperatures (in Celsius)
temperatures_celsius = np.linspace(25, 70, 100)
# Convert to Kelvin
temperatures_kelvin = temperatures_celsius + 273.15

# Impurity concentrations
concentrations = [0, 0.001, 0.01, 0.1]  # From 0 to 0.1 M (none, low, medium, high impurity levels)


# Plotting the results
plt.figure(figsize=(6, 4))

for c in concentrations:
    rates = r(temperatures_kelvin, c)
    plt.plot(temperatures_celsius, rates, label=f'Impurity concentration: {c:.4g} M')

# Set logarithmic scale for y-axis
plt.yscale('log')

# Adding labels and title
plt.xlabel('Temperature (°C)')
plt.ylabel('Reaction Rate (log scale, arbitrary units)')
plt.title('Effect of Temperature and Ion Impurities on Corrosion Rate')
plt.legend()
plt.grid(True)
plt.show()

# Print reaction rates at 25°C and 70°C for different concentrations
T_25C = 298.15  # 25°C in Kelvin
T_70C = 343.15  # 70°C in Kelvin

print("Reaction rates at 25°C and 70°C for different impurity concentrations:")
for c in concentrations:
    rate_25C = r(T_25C, c)
    rate_70C = r(T_70C, c)
    print(f"Concentration: {c:.4f} M - Rate at 25°C: {rate_25C:.2e}, Rate at 70°C: {rate_70C:.2e}")
