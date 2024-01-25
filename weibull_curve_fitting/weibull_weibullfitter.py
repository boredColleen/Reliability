import numpy as np
import matplotlib.pyplot as plt
from lifelines import WeibullFitter, KaplanMeierFitter

# Sample data
times = np.array([800, 1000, 1200])  # Your data
events = np.array([1, 1, 0])  # 1 if event observed, 0 if censored (assuming the last one is censored)

# Kaplan-Meier Fitter
kmf = KaplanMeierFitter()
kmf.fit(times, event_observed=events)
km_survival = kmf.survival_function_

# Weibull Fitter
wf = WeibullFitter()
wf.fit(times, event_observed=events)
weibull_survival = wf.survival_function_at_times(times)

# Extracting the shape and scale parameters
beta = wf.rho_
eta = wf.lambda_
print('beta, eta:', beta, eta)

# Time range for predictions
t = np.linspace(0, max(times)*2, 100)

# Plotting both survival curves
plt.figure(figsize=(10, 6))

# Kaplan-Meier plot
plt.step(km_survival.index, km_survival['KM_estimate'], where='post', label='Kaplan-Meier Estimate', linestyle='--', color='orange')

# Weibull plot
plt.plot(t, wf.survival_function_at_times(t), label='Weibull Survival', linestyle='-', color='blue')

# Scatter plot for events
plt.scatter(times[events == 1], km_survival.loc[times[events == 1]].values, label='Events (Kaplan-Meier)', color='purple', marker='D')

# Configure the plot
plt.title('Survival Curves Comparison (Kaplan-Meier vs Weibull)')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Print Weibull survival probabilities for each data point
for time, survival_prob in zip(times, weibull_survival):
    print(f"{survival_prob}")


