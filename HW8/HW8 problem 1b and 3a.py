from scipy.stats import nbinom
from scipy.stats import poisson

#1b
print(nbinom.sf(9, 19, 12/17))
print(poisson.sf(9, 7.5))

#3a
print(1 - poisson.cdf(10, 4.091))
print(1 - poisson.cdf(10, 12))