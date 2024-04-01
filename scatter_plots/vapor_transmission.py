import numpy as np
import matplotlib.pyplot as plt

# Constants
Ea = 0.7  # Activation energy in eV
k = 8.617e-5  # Boltzmann constant in eV/K

# Temperature in Celsius and Kelvin
temp_celsius = 85
temp_kelvin = temp_celsius + 273.15
temp_range_celsius = np.linspace(20, 90, 500)
temp_range_kelvin = temp_range_celsius + 273.15

# Vapor transmission amounts for the original tests
vapor_transmission = [1.4, 0.1]  # in grams

# Function to calculate vapor transmission curve
def calculate_vapor_transmission(T2, T1, vapor_transmission_at_T1):
    return vapor_transmission_at_T1 * np.exp(Ea / k * (1/T1 - 1/T2))

# Calculate vapor transmission for each curve and the enhanced curve
vapor_transmission_curve1 = calculate_vapor_transmission(temp_range_kelvin, temp_kelvin, vapor_transmission[0])
vapor_transmission_curve2 = calculate_vapor_transmission(temp_range_kelvin, temp_kelvin, vapor_transmission[1])
vapor_transmission_curve2_enhanced = vapor_transmission_curve2 * 20

# Specific point at 60°C for the enhanced curve
temp_specific_celsius = 60
temp_specific_kelvin = temp_specific_celsius + 273.15
vapor_transmission_specific = calculate_vapor_transmission(temp_specific_kelvin, temp_kelvin, vapor_transmission[1]) * 20

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter([temp_celsius, temp_celsius], vapor_transmission, color='red', label='Packaging Tests')
plt.plot(temp_range_celsius, vapor_transmission_curve1, 'r--', label='Vapor Transmission Curve 1')
plt.plot(temp_range_celsius, vapor_transmission_curve2, 'b--', label='Vapor Transmission Curve 2')
plt.plot(temp_range_celsius, vapor_transmission_curve2_enhanced, 'g-.', label='Enhanced Curve 2 (x20)')
plt.axhline(y=1, color='green', linestyle='--', linewidth=1, label='1g Reference Line')
plt.scatter([temp_specific_celsius], [vapor_transmission_specific], color='green', zorder=5, label='Point at 60°C (Enhanced Curve)')

# Labeling the plot
plt.xlabel('Temperature (°C)')
plt.ylabel('Vapor Transmission (g)')
plt.title('Vapor Transmission vs. Temperature')
plt.xlim(35, 95)
plt.ylim(-0.1, 2)
plt.legend()
plt.grid(True)

# Show plot
plt.show()
