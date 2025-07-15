# Bayesian Age-Period-Cohort (APC) Model in Python
This repository contains a Python script implementing a Bayesian Age-Period-Cohort (APC) model using PyMC. The model is designed for analyzing count data (e.g., disease incidence, demographic events) by estimating the separate effects of age, period, and cohort using a Poisson likelihood.

## Features
- Bayesian inference for age, period, and cohort effects using PyMC
- Synthetic data generation for demonstration
- MCMC sampling for posterior estimation
- Prints model component shapes for transparency and debugging

## Usage
1. Clone the repository or download the script.
2. Install required packages:
    pip install numpy pymc arviz pytensor
3. Run the script:
    python mcmc.py
The script will generate synthetic data, fit the APC model, and print the shapes of model components.

## Example Output
Shape of person_years: (15, 6)
Shape of age_effect: (15,)
Shape of period_effect: (6,)
Shape of cohort_effect: (20,)
Shape of cohort_indices: (15, 6)
Shape of log_mu: (15, 6)
Shape of mu: (15, 6)


## License
This project is licensed under the MIT License.

## Contact
For questions or suggestions, contact sahar.agarwal_ug2023@gmail.com


