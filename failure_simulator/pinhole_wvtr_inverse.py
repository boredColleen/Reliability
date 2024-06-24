import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import aesara.tensor as at
import scipy.stats as stats
import arviz as az

# Constants from previous results
shape_weibull = 5.43  # Weibull shape parameter obtained
scale_weibull = 20.5  # Weibull scale parameter obtained

# Known electrolyte distribution parameters
mean_electrolyte_effect = 1.0
std_dev_electrolyte_effect = 0.2

# Simulate some data for the purpose of this example
np.random.seed(42)
observed_data = np.random.weibull(shape_weibull, 3000) * scale_weibull  # Simulated data based on Weibull parameters

# Setup PyMC model
with pm.Model() as model:
    # Priors for unknown model parameter - increase in WVTR due to pinholes
    pinhole_effect = pm.Normal('pinhole_effect', mu=7, sigma=10)

    # Calculating the theoretical mean WVTR based on pinhole effect
    base_wvtr = 10
    wvtr_from_pinhole = pm.Deterministic('wvtr_from_pinhole', base_wvtr + pinhole_effect)

    # Electrolyte effect modeled as a normal distribution
    electrolyte_effect = pm.Normal('electrolyte_effect', mu=mean_electrolyte_effect, sigma=std_dev_electrolyte_effect)

    # Calculating the final WVTR after electrolyte effect
    final_wvtr = pm.Deterministic('final_wvtr', wvtr_from_pinhole * electrolyte_effect)

    # Likelihood (sampling distribution) of observed data
    observed_wvtr = pm.Weibull('observed_wvtr', alpha=shape_weibull, beta=final_wvtr, observed=observed_data)

    # Posterior distribution sampling using MCMC
    trace = pm.sample(3000, tune=1500, cores=1)  # Set cores to 1 to avoid multiprocessing issues

# Plot the posterior of the pinhole effect
az.plot_posterior(trace, var_names=['pinhole_effect'])
plt.show()

# Print the estimated mean pinhole effect
pinhole_effect_mean = trace.posterior['pinhole_effect'].mean().item()
print(f"Estimated mean pinhole effect: {pinhole_effect_mean:.4f}")
