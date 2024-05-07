import numpy as np
import matplotlib.pyplot as plt

def calculate_corrosion_time(j, C_I, T):
    """
    Calculate the time until a copper electrode fails due to corrosion based on current density in mA/cm^2, 
    impurity concentration, and temperature.
    """
    # Constants
    A = 1E6           # Frequency factor (s^-1)
    E_a = 0.7         # Activation energy (eV)
    k = 8.617e-5      # Boltzmann's constant (eV/K)
    thickness = 1e-6  # Thickness of copper electrode (m)
    density_Cu = 8.96 # Density of copper (g/cm^3)
    M_Cu = 63.546     # Molar mass of copper (g/mol)
    beta = 5          # Effect of current density on corrosion rate per mA/cm^2
    gamma = 10        # Effect of impurity concentration on corrosion rate
    
    # Convert temperature from Celsius to Kelvin
    T_K = T + 273.15

    # Convert current density from mA/cm^2 to A/cm^2 for calculation
    j_A_cm2 = j / 1000

    # Calculate the corrosion rate (mol/cm^2/s)
    corrosion_rate = A * np.exp(-E_a / (k * T_K)) * (1 + beta * j_A_cm2 + gamma * C_I)

    # Convert corrosion rate to mass loss rate (g/cm^2/s)
    mass_loss_rate = corrosion_rate * M_Cu

    # Calculate thickness loss rate (m/s)
    thickness_loss_rate = mass_loss_rate / (density_Cu * 1e6)

    # Calculate time until failure (s)
    time_until_failure = thickness / thickness_loss_rate

    # Convert time from seconds to hours
    time_until_failure_hours = time_until_failure / 3600

    return time_until_failure_hours

# Generate random data for current density, impurity concentration, and temperature
n_samples = 200
current_densities = np.random.randint(10, 1001, size=n_samples)
impurity_concentrations = 0.01 + np.random.rand(n_samples) * (1 - 0.01)
temperatures = np.random.choice(np.arange(25, 90, 5), size=n_samples)  # Choose temperatures from 25°C to 85°C at 5°C intervals

# Calculate times until failure
times_to_failure = [calculate_corrosion_time(j, C_I, T) for j, C_I, T in zip(current_densities, impurity_concentrations, temperatures)]

# Scatter plot of the data
plt.figure(figsize=(10, 6))
sc = plt.scatter(current_densities, impurity_concentrations, c=temperatures, cmap='viridis', vmin=25, vmax=85)
plt.colorbar(sc, label='Temperature (°C)')
plt.title('Scatter Plot of Corrosion Times with Temperature Color Coding')
plt.xlabel('Current Density (mA/cm²)')
plt.ylabel('Ionic Impurity Concentration (M)')
plt.grid(True)
plt.show()
