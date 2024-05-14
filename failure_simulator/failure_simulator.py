import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import pandas as pd

def weibull_survival(x, k, lam):
    """Calculate the Weibull survival function."""
    return np.exp(-(x / lam)**k)

def simulate_and_prepare_data(k, lam, num_samples, max_time):
    """Simulate failure events and prepare a unified dataset for Kaplan-Meier analysis."""
    # Generate random samples from the Weibull distribution
    samples = lam * np.random.weibull(k, num_samples)
    
    # Identify which samples are failures within the max_time and which are censored
    failure_times = samples[samples <= max_time]
    censored_times = np.full((samples > max_time).sum(), max_time)  # All censored at max_time
    
    # Combine into one dataset with event indicators, using -1 for censored data
    times = np.concatenate((failure_times, censored_times))
    events = np.concatenate((np.ones(len(failure_times)), -1 * np.ones(len(censored_times))))  # 1 for failures, -1 for censoring
    
    return times, events

def plot_kaplan_meier(times, events, k, lam):
    """Plot Kaplan-Meier survival curve from times and event indicators alongside the Weibull survival curve."""
    x_values = np.linspace(0, 2 * lam, 260)
    y_survival = weibull_survival(x_values, k, lam)
    
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=events == 1)  # Only consider events == 1 as failures
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_survival, label='Weibull Survival Function', color='blue')
    kmf.plot_survival_function(label='Kaplan-Meier Estimate', color='darkgreen')
    
    # Adding scatter for each point
    failures = times[events == 1]
    censored = times[events == -1]
    plt.scatter(failures, kmf.survival_function_at_times(failures), color='red', zorder=10, label='Failures')
    plt.scatter(censored, kmf.survival_function_at_times(censored), color='green', zorder=10, label='Censored')
    
    plt.title('Kaplan-Meier Survival Estimate with Failures and Censoring')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

def round_times_to_24hr_intervals(times):
    """Round times up to the nearest 24-hour interval."""
    return np.ceil(times / 24) * 24

def save_to_csv(times, events, filename):
    """Save the times and events data to a CSV file."""
    df = pd.DataFrame({
        'times': times,
        'events': events.astype(int)  # Cast events to integers
    })
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    
# Parameters
k = 2.0  # Shape parameter
lam = 208  # Scale parameter
max_time = 216  # Maximum time (hours) for considering events

times, events = simulate_and_prepare_data(k, lam, 9, max_time)
rounded_times = round_times_to_24hr_intervals(times)
save_to_csv(rounded_times, events, 'failure_simulator/times_and_events.csv')
plot_kaplan_meier(times, events, k, lam)
