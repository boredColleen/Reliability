import numpy as np

# Constants
k_B = 8.617e-5  # Boltzmann constant in eV/K
T = 358  # Temperature in K (85°C)
days = 30  # Duration of the simulation
samples = 20  # Number of samples
mean_humidity = 0.3  # Mean daily moisture uptake in g/day
std_humidity = 0.2  # Standard deviation of moisture uptake
A = 1e12  # Reduced frequency factor in 1/s (from 1e13)
E_a = 1.0  # Increased activation energy in eV (from 0.8 eV)
corrosion_threshold = 0.1  # Increased threshold for corrosion to consider as failure

def corrosion_rate(humidity, T):
    """Calculate corrosion rate based on moisture and temperature using Arrhenius equation."""
    R = k_B  # Use the Boltzmann constant for R
    rate = A * np.exp(-E_a / (R * T)) * humidity ** 0.5  # Reduced impact of humidity
    return rate

def simulate_corrosion(days, samples):
    """Simulate the corrosion process over a given number of days for multiple samples."""
    failure_count = np.zeros(days)  # Track number of failures each day
    corrosion_levels = np.zeros(samples)  # Initial corrosion levels for each sample

    for day in range(days):
        daily_humidity = np.random.normal(mean_humidity, std_humidity, samples)  # Generate daily humidity levels
        daily_corrosion = corrosion_rate(daily_humidity, T)  # Calculate daily corrosion rates
        corrosion_levels += daily_corrosion  # Accumulate corrosion levels
        
        # Check for failures
        failures_today = corrosion_levels >= corrosion_threshold
        failure_count[day] = np.sum(failures_today)  # Count failures for the day
        corrosion_levels[failures_today] = 0  # Reset corrosion levels for failed components

    return failure_count

# Execute simulation and print results
failure_results = simulate_corrosion(days, samples)
print("Daily failure count:", failure_results)
