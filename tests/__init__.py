"""
These are end-to-end tests that check that random samples from the joint distribution have:
(i) the expected rank correlation matrix
(ii) the expected marginal distributions (using the Kolmogorov-Smirnov statistic)

Some of this may also be covered in tests of copula functionality from the `statsmodels` package, but I don't have a
full understanding of what they do.

todo: consider if these tests, based on random sampling, could be replaced or supplemented by tests based on integration
todo: code could be cleaner
"""
