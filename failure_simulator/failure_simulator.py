import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def weibull_pdf(x, k, lam):
    """Calculate the Weibull PDF at x for given shape (k) and scale (lam) parameters."""
    return (k / lam) * (x / lam)**(k-1) * np.exp(-(x / lam)**k)

def weibull_survival(x, k, lam):
    """Calculate the Weibull survival function."""
    return np.exp(-(x / lam)**k)

def simulate_weibull_kaplan_meier(k, lam, num_samples):
    """Simulate Weibull distribution and determine failure based on Weibull PDF."""
    x_values = np.linspace(0, 2 * lam, 400)
    y_survival = weibull_survival(x_values, k, lam)
    
    # Generate random samples from the Weibull distribution
    samples = lam * np.random.weibull(k, num_samples)
    pdf_values = weibull_pdf(samples, k, lam)
    
    # Determine event occurrence based on the Weibull PDF and random comparison
    random_probs = np.random.uniform(0, max(pdf_values), num_samples)
    event_occurred = random_probs < pdf_values
    
    # Plotting the Weibull Survival Function
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_survival, label='Weibull Survival Function', color='blue')
    
    # Kaplan-Meier Estimator
    kmf = KaplanMeierFitter()
    kmf.fit(durations=samples, event_observed=event_occurred)
    
    # Plot the Kaplan-Meier Survival Function
    kmf.plot_survival_function(label='Kaplan-Meier Estimate', color='darkgreen')
    
    # Plotting event occurrences
    failure_times = samples[event_occurred]
    censored_times = samples[~event_occurred]
    plt.scatter(failure_times, kmf.survival_function_at_times(failure_times), color='red', zorder=10)
    plt.scatter(censored_times, kmf.survival_function_at_times(censored_times), color='green', zorder=10)
    
    plt.title('Weibull Survival Function vs. Kaplan-Meier Estimate')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters
k = 1.5  # Shape parameter
lam = 100  # Scale parameter

simulate_weibull_kaplan_meier(k, lam, 24)
