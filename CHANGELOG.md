# 1.9

- New function `local_correlation_dimension`.
- Functions that were already exported as public are now also in the documentation:
  `pointwise_dimensions, pointwise_correlationsums`

# 1.8

- New way to estimate extreme value theory dimensions based on block maxima.
- Giving `p::Real` to `extremevaltheory_...` functions is deprecated. Now the types `Exceedances` or `BlockMaxima` need to be instantiated and used instead.

# 1.7.1
- Change the indices of the outer sum of the correlationsum to i = 1 to N and adapt normalisations.

# 1.7

- Update `extremevaltheory_gpdfit_pvalues` to calculate further the NRMSE of the GPD fits.
- Set default estimator to `:exp` for EVT.
- Bugfixes in `extremevaltheory_gpdfit_pvalues`
- Add Cramer Von Mises estimator for `extremevaltheory_gpdfit_pvalues`

# 1.6

- new function `extremevaltheory_gpdfit_pvalues` that can help quantify the "goodness" of the results of the EVT dimension
- new probability weighted moments estimator for GPD fit to data used in the EVT fractal dimensions

# 1.5
- added an interface based on the `slopefit` function, that allows estimating the "linear scaling region" using an extendable API. The methods methods subtype `SlopeFit` and extend `_slopefit(x, y, t::SlopeFit, ci::Real)`.
- Added returning confidence intervals for all slope fitting methods
- The function `linear_region` is now deprecated/not-documented in favor of `slopefit(x, y, LargestLinearRegion())`

# 1.4

- Showing progress bars can be turned off module-wide by setting `ENV["FRACTALDIMENSIONS_PROGRESS"] = false`.
- New, faster multithreaded implementation for `correlationsum`.
- Generalized dimension is now also multithreaded and has a progress bar.
- `minimum_pairwise_distance` is now exported and also switches to a brute force search for high dimensional data.

# 1.3

- New function `pointwise_dimensions`.

# 1.2

- Added extreme value theory based estimators for local fractal dimension and persistence

# 1.1

- Massive performance boosts in correlation sum and box-assisted version
- Added and exported `prismdim_theiler`, and clarified documentation around prism dimension

# 1.0

Initial package release. Previously the code here was part of ChaosTools.jl.