import scipy.stats as stats
import numpy as np

### Task Group 1 ###
## Variable "lam" to represent the rate parameter of Poisson(7)
lam = 7

## Probability of observing 'lam' defects on a given day
prob_lam = stats.poisson.pmf(lam, lam)
print(prob_lam)

## Probability of observing less than or equal to (le) 4 defects on a given day
prob_le_4 = stats.poisson.cdf(4, lam)
print(prob_le_4)

## Probability of greater than (gt) 9 defects on a given day
prob_gt_9 = 1 - stats.poisson.cdf(9, lam)
print(prob_gt_9)


### Task Group 2 ###
## Random value set of 365 days with expected value of our Lambda 7
year_defects = stats.poisson.rvs(lam, size=365)

## First 20 values of randomized distribution dataset
print(year_defects[0:20])

## How many defects to expect over 365 day period with expected value (lam, mean) 
print(lam*365)

## Sum of randomized dataset of yearly defects
print(sum(year_defects))

## Average number of defects per day in similated dataset, avg is just below expected value of 7
print(np.mean(year_defects)) 

## Maximum number of defects per day in simulated dataset
print(max(year_defects))

## Probability of observing maximum value or more from Poisson(7)
prob_max_defect = 1 - stats.poisson.cdf(16, lam)
print(prob_max_defect)

### Extra Bonus ###
# Number of defects in 90th percentile for a given day
pct90_defects = stats.poisson.ppf(0.9, lam)
print(pct90_defects)

# Proportion of simulated dataset greater than or equal to (ge) number calculated in previous step
prop_ge_pct90_defects = sum(year_defects >= pct90_defects) / len(year_defects)
print(prop_ge_pct90_defects)