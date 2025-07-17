Alternatives:

1. Scale your distribution first, then transform back:

pythonimport numpy as np
from scipy import stats

# Instead of working with tiny ranges directly
# Scale up, compute, then scale back down
scale_factor = 1e6  # or whatever makes sense
scaled_dist = stats.your_dist(loc=0, scale=original_scale * scale_factor)
scaled_result = scaled_dist.ppf(probabilities)
actual_result = original_loc + scaled_result / scale_factor

2. Use higher precision arithmetic:

python# For extreme precision needs
import mpmath
mpmath.mp.dps = 50  # 50 decimal places

# Or use numpy's extended precision
probabilities = np.array(your_probs, dtype=np.longdouble)

3. Work in log-space when possible:

python# Use logpdf/logcdf and work backwards
log_probs = np.log(probabilities)
# Then use numerical methods to solve log(CDF(x)) = log_probs

4. Clamp probabilities away from extremes:

python# Keep probabilities well away from 0 and 1
probs = np.clip(probabilities, 1e-12, 1-1e-12)
results = distribution.ppf(probs)

5. Use interpolation for very small ranges:

python# Pre-compute a fine grid and interpolate
x_grid = np.linspace(dist.ppf(1e-10), dist.ppf(1-1e-10), 10000)
cdf_grid = dist.cdf(x_grid)
from scipy.interpolate import interp1d
ppf_interp = interp1d(cdf_grid, x_grid, bounds_error=False, fill_value='extrapolate')
results = ppf_interp(probabilities)