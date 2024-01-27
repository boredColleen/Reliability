## Weibull Model Fitting and Visualization

This script demonstrates the process of fitting a Weibull distribution model to both observed and censored survival data. The primary objectives are to estimate the shape and scale parameters of the Weibull distribution and to determine an optimal mix of survival functions derived from observed-only and full datasets (including both observed and censored data).

### Key Components

1. **Negative Log-Likelihood Functions**:
    - `neg_log_likelihood`: Defines the negative log-likelihood for the Weibull distribution, taking into account both observed and censored data. This function is used to estimate the shape and scale parameters of the distribution.
    - `neg_log_likelihood_theta`: Calculates the negative log-likelihood for a mixed model that combines survival functions from observed-only and full data. The mixing parameter `theta` determines the contribution of each model to the mixed survival function.

2. **Data Preparation**:
    - Sample times (`times`) and event indicators (`events`) are defined, with `1` indicating an observed event and `0` indicating a censored event.
    - Initial guesses for the shape and scale parameters are set.

3. **Parameter Estimation**:
    - `minimize` from `scipy.optimize` is used to find the parameters that minimize the negative log-likelihood.
    - The process is performed twice: first, considering observed data only; second, considering both observed and censored data.

4. **Survival Function Calculation**:
    - The failure rate based on observed events is calculated.
    - Survival functions for observed-only data (`sf_values_observed`) and full data (`sf_values_full`) are computed over a finely spaced time grid.
    - A mixed survival function (`sf_values_mixed`) is formed by combining `sf_values_observed` and `sf_values_full`, weighted by the parameter `theta`.

5. **Optimization of Theta**:
    - The `theta` parameter, which controls the mixing of survival functions, is optimized to best fit the observed data.
    - The optimal `theta` value is printed.

6. **Visualization**:
    - The mixed survival function is plotted against time.
    - The plot is properly labeled and displayed with a tight layout for clarity.

### Usage

Run the script to perform Weibull model fitting, optimize the mixing parameter `theta`, and visualize the mixed survival function. Ensure that the necessary libraries (`numpy`, `scipy`, and `matplotlib`) are installed.
