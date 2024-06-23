import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import scipy.stats as stats
import multiprocessing as mp

# Set the multiprocessing start method for macOS compatibility
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass  # If already set

# Observed Weibull parameters
shape_obs = 2.85
scale_obs = 11.65

model = pm.Model()

with model:
    # Define priors
    a = pm.Normal('a', mu=4.0, sigma=0.25)
    b = pm.Normal('b', mu=100, sigma=0.2)
    c = pm.Normal('c', mu=250, sigma=10)

    # Expected mean area and height
    area_mean = a
    height_mean = b
    water_ingress_mean = c * area_mean / height_mean

    # Weibull likelihood for the observed water ingress
    observed_water_ingress = pm.Weibull('obs', alpha=shape_obs, beta=scale_obs, observed=water_ingress_mean)

    # Sampling
    #trace = pm.sample(5000, tune=1000, progressbar=True, cores=1)
    trace = pm.sample(5000, tune=1000, target_accept=0.95, progressbar=True, cores=1)


# Extract the mean estimates
a_mean = np.mean(trace['a'])
b_mean = np.mean(trace['b'])
c_mean = np.mean(trace['c'])

# Generate synthetic water ingress data using the mean estimates
synthetic_areas = a_mean * np.random.rand(10000)
synthetic_heights = b_mean * np.random.rand(10000)

#
synthetic_water_ingress = c_mean * synthetic_areas / synthetic_heights

# Fit a Weibull distribution to the synthetic data
params = stats.weibull_min.fit(synthetic_water_ingress, floc=0)  # Fix location to zero

# Plotting observed vs. modeled distributions
x = np.linspace(min(synthetic_water_ingress), 5, 1000) #max(synthetic_water_ingress), 1000)
plt.figure(figsize=(10, 5))
plt.plot(x, stats.weibull_min.pdf(x, shape_obs, scale=scale_obs), 'r-', label='Observed Weibull Distribution')
plt.plot(x, stats.weibull_min.pdf(x, *params), 'b--', label='Modeled Weibull Distribution')
plt.title('Comparison of Observed and Modeled Weibull Distributions')
plt.xlabel('Water Ingress')
plt.ylabel('Density')
plt.legend()
plt.show()

# Print the estimated parameters
print(f"Estimated mean of a: {a_mean:.2f}")
print(f"Estimated mean of b: {b_mean:.2f}")
print(f"Estimated mean of c: {c_mean:.2f}")
