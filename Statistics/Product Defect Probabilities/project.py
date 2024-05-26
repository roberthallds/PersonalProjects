import scipy.stats as stats
import numpy as np

### Task Group 1 ###
## Task 1: Create a variable 'lam' representing the rate parameter of the distribution
lam = 7

## Task 2: Calculate and print the probability of observing exactly lam = 7 defects on a given day
print(stats.poisson.pmf(7, 7))

## Task 3: Calculate and print the probability of having a day of <= 4 defects (an exceptionally good daily rate)
lessThan4 = stats.poisson.cdf(4, 7)
print(lessThan4)

## Task 4: Calculate and print the probability of having a day of >= 9 defects (a poor daily rate)
moreThan9 = 1 - stats.poisson.cdf(9, 7)
print(moreThan9)

### Task Group 2 ###

## Task 5: Create a variable yearDefects having 365 random variables from the Poisson distribution
yearDefects = stats.poisson.rvs(7, size=365)

## Task 6:
print(yearDefects[0:20])

## Task 7: Calculate and print the total number of expected defects over 365 days if the expected value is 7
print(7 * 365)

## Task 8: Calculate and print the total sum of the dataset yearDefects, and compare to the number of defects expected over 365 days
print(yearDefects.sum())

## Task 9: Calculate and print the average number of defects per day from simulated dataset, and compare to the lambda expected value of 7
print(yearDefects.mean())

## Task 10: Calculate and print the maximum value of yearDefects
print(yearDefects.max())

## Task 11: Calculate and print the probability of observing the maximum value or more from the Poisson Distribution
print(1 - stats.poisson.cdf(15, 7))

### Extra Bonus ###

# Task 12: Calculate and print the number of defects that would put production in the 90th percentile of defectives for a given day
percentile90 = stats.poisson.ppf(0.9, 7)
print(percentile90)

# Task 13: Calculate and print the proportion of simulated dataset is greater than or equal to the number in the previous step
print(sum(yearDefects >= percentile90) / len(yearDefects))