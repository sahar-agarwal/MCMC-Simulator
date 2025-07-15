import numpy as np
import pymc as pm
import arviz as az
import pytensor

# Generating data
np.random.seed(42)
age_groups = 15
periods = 6
person_years = np.random.poisson(100, size = (age_groups, periods))
cases = np.random.poisson(lam = 10, size = (age_groups, periods))

def compute_shape(symbolic_var):
    shape_func = pytensor.function([], symbolic_var.shape)
    return shape_func()

def apc_model(cases, person_years):
  with pm.Model() as model:
    age_effect = pm.Normal("age_effect", mu = 0, sigma = 1, shape = age_groups) # age_effect ~ N(0, 1)
    period_effect = pm.Normal("period_effect", mu = 0, sigma = 1, shape = periods) # period_effect ~ N(0, 1)
    n_cohorts = age_groups + periods - 1
    cohort_effect = pm.Normal("cohort_effect", mu = 0, sigma = 1, shape=n_cohorts) # cohort_effect ~ N(0, 1)
    cohort_indices = np.add.outer(np.arange(age_groups), np.arange(periods)) # age and period combination indices

    log_mu = (
        np.log(person_years)
        + age_effect[:, None]
        + period_effect[None, :]
        + cohort_effect[cohort_indices]
    )
    mu = pm.math.exp(log_mu)
    cases_observed = pm.Poisson("cases_observed", mu = mu, observed = cases) # likelihood f(x) assuming a poisson distribution is followed: cases ~ P(mu)

    trace = pm.sample(2000, tune = 1000, target_accept = 0.95, random_seed = 42) # 2000 samples, 1000 tuning iterations

    print("Shape of person_years:", person_years.shape)
    print("Shape of age_effect:", compute_shape(age_effect))
    print("Shape of period_effect:", compute_shape(period_effect))
    print("Shape of cohort_effect:", compute_shape(cohort_effect))
    print("Shape of cohort_indices:", cohort_indices.shape)
    print("Shape of log_mu:", compute_shape(log_mu))
    print("Shape of mu:", compute_shape(mu))

  return model, trace

model, trace = apc_model(cases, person_years)
