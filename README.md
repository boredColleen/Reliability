# Weibull Model Fitting from Scratch

## Overview
This document summarizes the process of fitting a Weibull model to survival data using Python. The process includes defining a likelihood function for the Weibull distribution, considering both censored and uncensored data, and using an optimization algorithm to estimate the parameters that maximize the likelihood function.

## Data Description
The data consists of time-to-event information, with some events being right-censored, meaning that the event had not occurred at the last observation time for those data points.

## Methodology

### Step 1: Define the Negative Log-Likelihood Function
The negative log-likelihood function for the Weibull distribution was defined considering both uncensored and censored data. For uncensored data, the likelihood contribution is based on the probability density function (PDF) of the Weibull distribution. For censored data, the likelihood contribution is based on the survival function (SF) of the Weibull distribution.

### Step 2: Parameter Estimation Using MLE
The `scipy.optimize.minimize` function was used to perform the maximization of the likelihood function (or minimization of the negative log-likelihood). Initial guesses were provided, and the L-BFGS-B method was used due to its efficiency and ability to handle bound constraints.

### Step 3: Calculation of PDF and SF
After obtaining the estimated shape and scale parameters from the optimization process, the Weibull PDF and SF were calculated for a fine grid of time points, spanning from 0 to just beyond the maximum time in the data.

### Step 4: Plotting the Results
The estimated Weibull PDF and SF were plotted using `matplotlib`. The actual data points were also plotted on the respective curves to visually assess the fit of the model. The data points were differentiated by color to indicate whether they were uncensored (red) or censored (green).

## Conclusion
The Weibull model fitting process allowed for an estimation of the shape and scale parameters that best represent the observed survival data. The resulting PDF and SF plots provided a visual tool for examining the fit of the model to the actual data.

---

Please note that the above content is a summary and does not include the specific Python code or mathematical equations used in the process. For the full implementation details, refer to the actual Python scripts.
