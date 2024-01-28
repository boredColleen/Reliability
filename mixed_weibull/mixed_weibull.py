import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from lifelines import KaplanMeierFitter  # Import the Kaplan-Meier fitter
import matplotlib.pyplot as plt

# Define the negative log-likelihood for the Weibull distribution
def neg_log_likelihood(params, times, events, include_censored):
    shape, scale = params
    uncensored_likelihood = np.where(events == 1, 
                                     np.log(shape) - np.log(scale) + (shape - 1) * (np.log(times) - np.log(scale)) - (times / scale) ** shape, 
                                     0)
    censored_likelihood = np.where(events == 0, 
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
    plt.plot(time_grid, sf_mixed, color='green')
    plt.step(kmf_survival.index, kmf_survival['KM_estimate'], where='post', color='blue', linestyle='--', linewidth=0.5)
    plt.scatter(times[events == 1], kmf_survival.loc[times[events == 1]], color='red', zorder=5, label='Observed Data')
    plt.scatter(times[events == 0], kmf_survival.loc[times[events == 0]], color='green', zorder=5, label='Censored Data')
    plt.xlabel('Time (h)')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# Sample data
times = np.array([80, 100, 430] + [504] * 20)
events = np.array([1, 1, 1] + [0] * 20)
initial_guess = [1.0, 1000.0]

# Minimize negative log-likelihood for observed and full data
result_observed = minimize(neg_log_likelihood, initial_guess, args=(times, events, False), bounds=[(0.001, None), (0.001, None)])
result_full = minimize(neg_log_likelihood, initial_guess, args=(times, events, True), bounds=[(0.001, None), (0.001, None)])
shape_observed, scale_observed = result_observed.x
shape_full, scale_full = result_full.x

# Calculate failure rate and survival functions
fr = np.mean(events)
time_grid = np.linspace(0, max(times) + 200, 1000)
sf_observed = fr * np.exp(-(time_grid / scale_observed) ** shape_observed) + (1 - fr)
sf_full = np.exp(-(time_grid / scale_full) ** shape_full)

# Minimize negative log-likelihood for theta
initial_guess_theta = [0.5]
result_theta = minimize(neg_log_likelihood_theta, initial_guess_theta, args=(time_grid, sf_full, sf_observed, times, events), bounds=[(0, 1)])
theta = result_theta.x[0]
print(f"fr:{fr}, a_o:{shape_observed}, b_o:{scale_observed}, a_f:{shape_full}, b_f:{scale_full}, theta:{theta}")
print(f"sf_observed = fr * exp(-(t/a_o^b_o) + (1-fr)")
print(f"sf_full = exp(-(t/a_f^b_f))")
print(f"sf_mixed = theta * sf_observed + (1- theta) sf_full")

# Plot survival function
sf_mixed = (1 - theta) * sf_full + theta * sf_observed

# Compute the Kaplan-Meier survival function
kmf = KaplanMeierFitter()
kmf.fit(times, event_observed=events)
kmf_survival = kmf.survival_function_

# Plot survival functions and scatter plot for observed and censored data
plot_survival_functions(time_grid, sf_mixed, kmf_survival, times, events)