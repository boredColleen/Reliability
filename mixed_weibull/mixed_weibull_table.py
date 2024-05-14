import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from lifelines import KaplanMeierFitter  # Import the Kaplan-Meier fitter
import matplotlib.pyplot as plt
import pandas as pd

# Define the negative log-likelihood for the Weibull distribution
def neg_log_likelihood(params, times, events, include_censored):
    shape, scale = params
    uncensored_likelihood = np.where(events == 1, 
                                     np.log(shape) - np.log(scale) + (shape - 1) * (np.log(times) - np.log(scale)) - (times / scale) ** shape, 
                                     0)
    censored_likelihood = np.where(events == -1, 
                                   - (times / scale) ** shape, 
                                   0)
    return -np.sum(uncensored_likelihood + censored_likelihood * include_censored)

# Define the likelihood function for theta
def neg_log_likelihood_theta(theta, time_grid, sf_full, sf_observed, times, events):
    sf_mixed = (1 - theta) * sf_full + theta * sf_observed
    pdf_values_mixed = -np.gradient(sf_mixed, time_grid)
    pdf_interp = interp1d(time_grid, pdf_values_mixed, kind='linear', fill_value='extrapolate')
    pdf_values_at_observed = pdf_interp(times[events == 1])
    return -np.sum(np.log(pdf_values_at_observed))
    
def plot_survival_functions(time_grid, sf_mixed, kmf_survival, times, events):
    """Plots survival functions and scatter plot for observed and censored data."""
    plt.figure(figsize=(5, 3))
    
    # Plot the mixed survival function
    plt.plot(time_grid, sf_mixed, color='green')
    
    # Plot the Kaplan-Meier estimate
    plt.step(kmf_survival.index, kmf_survival['KM_estimate'], where='post', color='blue', linestyle='--', linewidth=0.5)
    
    # Scatter plot for observed and censored data
    plt.scatter(times[events == 1], kmf_survival.loc[times[events == 1]], color='red', zorder=5, label='Failed')
    plt.scatter(times[events == -1], kmf_survival.loc[times[events == -1]], color='green', zorder=5, label='Survived')
    
    # Calculate the derivative of the survival function (approximate)
    derivative = -np.diff(sf_mixed) / np.diff(time_grid)
    
    # Find the time point where the derivative is minimized (least steep slope)
    min_decline_idx = np.argmax(derivative)
    min_decline_time = time_grid[min_decline_idx]
    min_decline_value = sf_mixed[min_decline_idx]
    min_derivative_value = derivative[min_decline_idx]
    
     # Calculate the slope in units of days (24 hours per day)
    derivative_per_day = min_derivative_value * 24 * 100 # percent per day
    print(f"min rate:{min_derivative_value}")
    # Annotate the point of minimum decline on the plot with a vertical line "|"
    plt.axvline(x=min_decline_time, color='red', linestyle='-.', label=f'{derivative_per_day:.2f} %/day')
    plt.annotate(f'{min_decline_time:.1f} h', (min_decline_time + 10, min_decline_value), textcoords="offset points", xytext=(0,10), ha='left', color='red')

    # Set labels and title
    plt.xlabel('Time (h)')
    plt.ylabel('Survival Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

#data = pd.read_csv('mixed_weibull/corrosion_multi_plot_data.csv', skipinitialspace=True)
data = pd.read_csv('failure_simulator/times_and_events.csv', skipinitialspace=True)

# Event column
#col = 'd'
col = 'events'

# Filter out events marked as '0' directly during load or preprocessing
data = data[data[col] != 0]

# Use numpy to repeat times based on the absolute value of events
times = np.repeat(data['times'], np.abs(data[col]))

# Determine signs for events (1 for positive, -1 for negative) and repeat accordingly
events = np.sign(data[col]).repeat(np.abs(data[col]))

# Initial guess for the Weibull parameters (shape, scale)
initial_guess = [1.0, 10.0]

# Minimize negative log-likelihood for observed and full data
result_observed = minimize(neg_log_likelihood, initial_guess, args=(times, events, False), bounds=[(0.001, None), (0.001, None)])
result_full = minimize(neg_log_likelihood, initial_guess, args=(times, events, True), bounds=[(0.001, None), (0.001, None)])
shape_observed, scale_observed = result_observed.x
shape_full, scale_full = result_full.x

# Calculate failure rate and survival functions
total_failures = np.sum(events[events > 0])  # Sum only positive values as they represent failures
total_censored = np.abs(np.sum(events[events < 0]))  # Sum the absolute values of negative numbers
total_events = total_failures + total_censored  # Total number of events, both failures and censored

failure_rate = total_failures / total_events if total_events > 0 else 0  # Avoid division by zero

time_grid = np.linspace(0, max(times)*1.2, 1000)
sf_observed = failure_rate * np.exp(-(time_grid / scale_observed) ** shape_observed) + (1 - failure_rate)
sf_full = np.exp(-(time_grid / scale_full) ** shape_full)

# Minimize negative log-likelihood for theta
initial_guess_theta = [0.5]
result_theta = minimize(neg_log_likelihood_theta, initial_guess_theta, args=(time_grid, sf_full, sf_observed, times, events), bounds=[(0, 1)])
theta = result_theta.x[0]
print(f"failure_rate:{failure_rate}")
print(f"scale_obs:{scale_observed},shape_obs:{shape_observed}")
print(f"scale_full:{scale_full}, shape_full:{shape_full}")
print(f"theta:{theta}")
print(f"sf_observed = fr * exp(-(t/a_o^b_o) + (1-fr)")
print(f"sf_full = exp(-(t/a_f^b_f))")
print(f"sf_mixed = theta * sf_observed + (1- theta) sf_full")

# Plot survival function
sf_mixed = (1 - theta) * sf_full + theta * sf_observed

# Prepare the data: converting -1 in 'events' to 0 for Kaplan-Meier compatibility
km_events = np.where(events == -1, 0, events)
# Compute the Kaplan-Meier survival function
kmf = KaplanMeierFitter()
kmf.fit(times, event_observed=km_events)
kmf_survival = kmf.survival_function_

# Plot survival functions and scatter plot for observed and censored data
plot_survival_functions(time_grid, sf_mixed, kmf_survival, times, events)
