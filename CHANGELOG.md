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