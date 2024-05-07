import numpy as np

def calculate_corrosion_time(j, C_I, T):
    """
    Calculate the time until a copper electrode fails due to corrosion based on current density, impurity concentration, and temperature.

    Parameters:
    j : float
        Current density (A/cm^2)
    C_I : float
        Concentration of ionic impurities (M)
    T : float
        Temperature (Celsius)

    Returns:
    float
        Time until failure (hours)
    """
    # Constants
    A = 1E3          # Frequency factor (s^-1)
    E_a = 0.5         # Activation energy (eV)
    k = 8.617e-5      # Boltzmann's constant (eV/K)
    thickness = 1e-6  # Thickness of copper electrode (m)
    density_Cu = 8.96 # Density of copper (g/cm^3)
    M_Cu = 63.546     # Molar mass of copper (g/mol)
    beta = 50       # Effect of current density on corrosion rate
    gamma = 10       # Effect of impurity concentration on corrosion rate
    
    # Convert temperature from Celsius to Kelvin
    T_K = T + 273.15

    # Calculate the corrosion rate (mol/cm^2/s)
    corrosion_rate = A * np.exp(-E_a / (k * T_K)) * (1 + beta * j + gamma * C_I)

    # Convert corrosion rate to mass loss rate (g/cm^2/s)
    mass_loss_rate = corrosion_rate * M_Cu

    # Calculate thickness loss rate (m/s)
    thickness_loss_rate = mass_loss_rate / (density_Cu * 1e6)  # convert g/cm^3 to g/m^3

    # Calculate time until failure (s)
    time_until_failure = thickness / thickness_loss_rate

    # Convert time from seconds to hours
    time_until_failure_hours = time_until_failure / 3600

    return time_until_failure_hours

# Example usage:
j = 0.0001       # Current density (A/cm^2)
C_I = 0.01     # Ionic impurity concentration (M)
T = 85         # Temperature (Celsius)

time_to_failure = calculate_corrosion_time(j, C_I, T)
print(f"Time until failure: {time_to_failure:.2f} hours")
