import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import scipy
    from scipy.optimize import fsolve
    from scipy.interpolate import interp1d

    return interp1d, plt, scipy


@app.cell
def _(scipy):
    """Functions implementing the PERT and modified PERT distributions.

    This module contains the core mathematical functions used by the PERT and modified PERT distribution
    classes. Each function takes the distribution parameters (minimum, mode, maximum, and optionally
    lambda) and implementsa specific statistical operation like pdf, cdf, etc.
    """

    import numpy as np

    def calc_alpha_beta(mini, mode, maxi, lambd):
        """Calculate alpha and beta parameters for the underlying beta distribution.

        Args:
            mini: Minimum value (must be < mode).
            mode: Most likely value (must be mini < mode < maxi).
            maxi: Maximum value (must be > mode).
            lambd: Shape parameter (must be > 0, typically 2-6 for practical applications).

        Returns:
            tuple[float, float]: Shape parameters alpha and beta for the beta distribution.

        """
        alpha = 1 + ((mode - mini) * lambd) / (maxi - mini)
        beta = 1 + ((maxi - mode) * lambd) / (maxi - mini)
        return alpha, beta
    _calc_alpha_beta = calc_alpha_beta
    def pdf(x, mini, mode, maxi, lambd=4):
        alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
        return scipy.stats.beta.pdf((x - mini) / (maxi - mini), alpha, beta) / (
            maxi - mini
        )

    def cdf(x, mini, mode, maxi, lambd=4):
        alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
        return scipy.stats.beta.cdf((x - mini) / (maxi - mini), alpha, beta)

    def sf(x, mini, mode, maxi, lambd=4):
        alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
        return scipy.stats.beta.sf((x - mini) / (maxi - mini), alpha, beta)

    def ppf(q, mini, mode, maxi, lambd=4):
        alpha, beta = calc_alpha_beta(mini, mode, maxi, lambd)
        return mini + (maxi - mini) * scipy.stats.beta.ppf(q, alpha, beta)

    def isf(q, mini, mode, maxi, lambd=4):
        alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
        return mini + (maxi - mini) * scipy.stats.beta.isf(q, alpha, beta)

    def rvs(mini, mode, maxi, lambd=4, size=None, random_state=None):
        alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
        return mini + (maxi - mini) * scipy.stats.beta.rvs(
            alpha,
            beta,
            size=size,
            random_state=random_state,
        )

    def mean(mini, mode, maxi, lambd=4):
        """Calculate the mean of the (modified) PERT distribution.

        This formula is equivalent to the traditional PERT mean formula
        (minimum + 4 * mode + maximum) / 6 when lambd=4.

        For the general case: μ = (mini + maxi + lambd * mode) / (2 + lambd)
        """
        return (maxi + mini + mode * lambd) / (2 + lambd)

    def var(mini, mode, maxi, lambd=4):
        """Calculate the variance of the (modified) PERT distribution.

        Uses the beta distribution variance formula: αβ/[(α+β)²(α+β+1)]
        transformed to PERT parameters using: var_pert = var_beta * (maxi - mini)²
        """
        alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)

        # Beta distribution variance: αβ/[(α+β)²(α+β+1)]
        beta_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

        # Transform to PERT scale
        return beta_var * (maxi - mini) ** 2

    def skew(mini, mode, maxi, lambd=4):
        numerator = 2 * (-2 * mode + maxi + mini) * lambd * np.sqrt(3 + lambd)
        denominator_left = 4 + lambd
        denominator_middle = np.sqrt(maxi - mini - mode * lambd + maxi * lambd)
        denominator_right = np.sqrt(maxi + mode * lambd - mini * (1 + lambd))
        denominator = denominator_left * denominator_middle * denominator_right
        return numerator / denominator

    def kurtosis(mini, mode, maxi, lambd=4):
        """Calculate the excess kurtosis of the (modified) PERT distribution.

        Uses the beta distribution kurtosis formula transformed to PERT parameters.
        Excess kurtosis = 6[(α-β)²(α+β+1) - αβ(α+β+2)] / [αβ(α+β+2)(α+β+3)]
        """
        alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)

        numerator = 6 * (
            (alpha - beta) ** 2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)
        )
        denominator = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)

        return numerator / denominator

    def stats(mini, mode, maxi, lambd=4):
        """Return the first four moments of the (modified) PERT distribution."""
        return (
            mean(mini, mode, maxi, lambd),
            var(mini, mode, maxi, lambd),
            skew(mini, mode, maxi, lambd),
            kurtosis(mini, mode, maxi, lambd),
        )

    def argcheck(mini, mode, maxi, lambd=4):
        return mini < mode < maxi and lambd > 0

    def get_support(mini, mode, maxi, lambd=4):
        """SciPy requires this per the documentation:

        If either of the endpoints of the support do depend on the shape parameters, then i) the
        distribution must implement the _get_support method; ...
        """
        return mini, maxi

    return calc_alpha_beta, np, ppf, stats


@app.cell
def _(funcs, scipy):
    """Arbitrary parameters in SciPy's ``rv_continuous`` class must be 'shape' parameters.
    Optional shape parameters are not supported, and are seemingly impossible to implement
    without egregious hacks. So there are two classes, one for the PERT distribution
    (with ``lambd=4``) and one for the modified PERT distribution (with ``lambd`` as a shape parameter).
    Beyond being repetitious, this also adversely affects the user-facing API.
    """

    class PERT(scipy.stats.rv_continuous):
        """The `PERT distribution <https://en.wikipedia.org/wiki/PERT_distribution>`_ is defined by the
        minimum, most likely, and maximum values that a variable can take. It is commonly used to
        elicit subjective beliefs. PERT is an alternative to the triangular distribution, but has a
        smoother shape.

        :param mini: The left bound of the distribution.
        :param mode: The mode of the distribution.
        :param maxi: The right bound of the distribution.


        Examples
        --------
        >>> from betapert import pert
        >>> dist = pert(0, 3, 12)
        >>> dist.mean()
        np.float64(4.0)
        >>> dist.cdf(5)
        np.float64(0.691229423868313)

        Equivalent to:

        >>> from betapert import pert
        >>> pert.cdf(5, 0, 3, 12)
        np.float64(0.691229423868313)

        """

        def _get_support(self, mini, mode, maxi):
            return funcs.get_support(mini, mode, maxi)

        def _argcheck(self, mini, mode, maxi):
            return funcs.argcheck(mini, mode, maxi)

        def _pdf(self, x, mini, mode, maxi):
            return funcs.pdf(x, mini, mode, maxi)

        def _cdf(self, x, mini, mode, maxi):
            return funcs.cdf(x, mini, mode, maxi)

        def _sf(self, x, mini, mode, maxi):
            return funcs.sf(x, mini, mode, maxi)

        def _isf(self, x, mini, mode, maxi):
            return funcs.isf(x, mini, mode, maxi)

        def _stats(self, mini, mode, maxi):
            return funcs.stats(mini, mode, maxi)

        def _ppf(self, q, mini, mode, maxi):
            return funcs.ppf(q, mini, mode, maxi)

        def _rvs(self, mini, mode, maxi, size=None, random_state=None):
            return funcs.rvs(mini, mode, maxi, size=size, random_state=random_state)

    class ModifiedPERT(scipy.stats.rv_continuous):
        """The modified PERT distribution generalizes the PERT distribution by adding a fourth parameter
        ``lambd`` that controls how much weight is given to the mode. ``lambd=4`` corresponds to the
        traditional PERT distribution.

        :param mini: The left bound of the distribution.
        :param mode: The mode of the distribution.
        :param maxi: The right bound of the distribution.
        :param lambd:
            The weight given to the mode. Relative to the PERT, values ``lambd < 4`` have the effect of
            flattening the density curve.


        Examples
        --------
        >>> from betapert import mpert
        >>> mdist = mpert(0, 3, 12, lambd=2)
        >>> mdist.mean()
        np.float64(4.5)

        Values of ``lambd<4`` have the effect of flattening the density curve

        >>> dist = mpert(0, 3, 12, lambd=4)
        >>> 1 - mdist.cdf(8), 1 - dist.cdf(8)
        (np.float64(0.11395114580927845), np.float64(0.04526748971193417))

        """

        def _get_support(self, mini, mode, maxi, lambd):
            return funcs.get_support(mini, mode, maxi, lambd)

        def _argcheck(self, mini, mode, maxi, lambd):
            return funcs.argcheck(mini, mode, maxi, lambd)

        def _pdf(self, x, mini, mode, maxi, lambd):
            return funcs.pdf(x, mini, mode, maxi, lambd)

        def _cdf(self, x, mini, mode, maxi, lambd):
            return funcs.cdf(x, mini, mode, maxi, lambd)

        def _sf(self, x, mini, mode, maxi, lambd):
            return funcs.sf(x, mini, mode, maxi, lambd)

        def _isf(self, x, mini, mode, maxi, lambd):
            return funcs.isf(x, mini, mode, maxi, lambd)

        def _stats(self, mini, mode, maxi, lambd):
            return funcs.stats(mini, mode, maxi, lambd)

        def _ppf(self, q, mini, mode, maxi, lambd):
            return funcs.ppf(q, mini, mode, maxi, lambd)

        def _rvs(self, mini, mode, maxi, lambd, size=None, random_state=None):
            return funcs.rvs(
                mini, mode, maxi, lambd, size=size, random_state=random_state
            )

    # ``pert`` and ``mpert`` being instances, not classes, is not IMO idiomatic Python, but it is core
    # to the way SciPy's ``rv_continuous`` class works. See examples of how SciPy defines their
    # distributions in ``scipy/stats/_continuous_distns.py``.
    pert = PERT()
    mpert = ModifiedPERT()
    return


@app.cell
def _(calc_alpha_beta, scipy):
    """Alternative PPF implementation #1: Scale distribution first, then transform back"""

    def ppf_scaled(q, mini, mode, maxi, lambd=4, scale_factor=1e6):
        """Scale up the distribution, compute PPF, then scale back down"""
        # Scale the parameters
        scaled_range = maxi - mini
        scaled_mini = 0
        scaled_mode = (mode - mini) * scale_factor / scaled_range
        scaled_maxi = scale_factor

        # Compute PPF on scaled distribution
        alpha, beta = calc_alpha_beta(
            scaled_mini, scaled_mode, scaled_maxi, lambd
        )
        scaled_result = scaled_mini + (
            scaled_maxi - scaled_mini
        ) * scipy.stats.beta.ppf(q, alpha, beta)

        # Transform back to original scale
        actual_result = mini + (scaled_result / scale_factor) * scaled_range
        return actual_result

    return (ppf_scaled,)


@app.cell
def _(np, scipy):
    """Alternative PPF implementation #2: Use higher precision arithmetic"""

    def ppf_high_precision(q, mini, mode, maxi, lambd=4):
        """Use numpy's extended precision for better numerical stability"""
        # Convert to extended precision
        q_extended = np.array(q, dtype=np.longdouble)
        mini_extended = np.longdouble(mini)
        mode_extended = np.longdouble(mode)
        maxi_extended = np.longdouble(maxi)
        lambd_extended = np.longdouble(lambd)

        # Calculate alpha and beta with extended precision
        alpha = 1 + ((mode_extended - mini_extended) * lambd_extended) / (
            maxi_extended - mini_extended
        )
        beta = 1 + ((maxi_extended - mode_extended) * lambd_extended) / (
            maxi_extended - mini_extended
        )

        # Compute PPF with extended precision
        result = mini_extended + (maxi_extended - mini_extended) * scipy.stats.beta.ppf(
            q_extended, alpha, beta
        )

        # Convert back to regular precision
        return np.array(result, dtype=np.float64)

    return (ppf_high_precision,)


@app.cell
def _(calc_alpha_beta, np, scipy, stats):
    """Alternative PPF implementation #3: Work in log-space when possible"""

    def ppf_log_space(q, mini, mode, maxi, lambd=4):
        """Use log-space to avoid numerical issues with extreme probabilities"""
        alpha, beta = calc_alpha_beta(mini, mode, maxi, lambd)
    
        # Handle scalar and array inputs consistently
        q = np.atleast_1d(q)
        results = np.zeros_like(q, dtype=float)
    
        for i, qi in enumerate(q):
            # Clamp to avoid log(0) or log(1)
            qi_safe = np.clip(qi, 1e-15, 1 - 1e-15)
            log_qi = np.log(qi_safe)
        
            # Define the equation to solve: log(CDF(x)) - log(q) = 0
            def log_cdf_eq(x_normalized):
                # Ensure x_normalized stays in [0,1]
                x_clamped = np.clip(x_normalized, 1e-15, 1 - 1e-15)
                return scipy.stats.beta.logcdf(x_clamped, alpha, beta) - log_qi
        
            try:
                # Use brentq with bounds instead of fsolve for more stability
                x_normalized = scipy.optimize.brentq(log_cdf_eq, 1e-10, 1 - 1e-10)
                results[i] = mini + (maxi - mini) * x_normalized
            
            except (ValueError, RuntimeError):
                # Fallback to regular ppf if log-space fails
                try:
                    x_normalized = stats.beta.ppf(qi_safe, alpha, beta)
                    results[i] = mini + (maxi - mini) * x_normalized
                except:
                    # Final fallback: linear interpolation
                    results[i] = mini + qi * (maxi - mini)
    
        return results[0] if len(results) == 1 else results

    return (ppf_log_space,)


@app.cell
def _(np, ppf):
    """Alternative PPF implementation #4: Clamp probabilities away from extremes"""

    def ppf_clamped(q, mini, mode, maxi, lambd=4, eps=1e-12):
        """Keep probabilities well away from 0 and 1"""
        # Clamp probabilities to avoid numerical issues
        q_clamped = np.clip(q, eps, 1 - eps)

        # Use regular PPF with clamped probabilities
        return ppf(q_clamped, mini, mode, maxi, lambd)

    return (ppf_clamped,)


@app.cell
def _(calc_alpha_beta, interp1d, np, scipy):
    """Alternative PPF implementation #5: Use interpolation for very small ranges"""

    def ppf_interpolated(q, mini, mode, maxi, lambd=4, grid_size=10000):
        """Pre-compute a fine grid and interpolate"""
        alpha, beta = calc_alpha_beta(mini, mode, maxi, lambd)

        # Create fine grid in probability space
        prob_grid = np.linspace(1e-10, 1 - 1e-10, grid_size)

        # Compute corresponding x values
        x_grid = mini + (maxi - mini) * scipy.stats.beta.ppf(prob_grid, alpha, beta)

        # Create interpolation function
        ppf_interp = interp1d(
            prob_grid, x_grid, bounds_error=False, fill_value="extrapolate"
        )

        # Interpolate for given probabilities
        return ppf_interp(q)

    return (ppf_interpolated,)


@app.cell
def _():
    import time
    return (time,)


@app.cell
def _(
    np,
    ppf,
    ppf_clamped,
    ppf_high_precision,
    ppf_interpolated,
    ppf_log_space,
    ppf_scaled,
    time,
):
    # Parameters
    params = {"mini": 0.3, "mode": 0.5, "maxi": 0.7}
    n_samples = 100000

    # Generate uniform random probabilities
    np.random.seed(42)  # For reproducibility
    probabilities = np.random.uniform(0, 1, n_samples)

    # Collect samples from each PPF implementation
    start_time = time.time()
    samples_original = np.array([ppf(p, **params) for p in probabilities])
    orig_time = time.time()
    samples_scaled = np.array([ppf_scaled(p, **params) for p in probabilities])
    scaled_time = time.time()
    samples_high_precision = ppf_high_precision(probabilities, **params)
    hp_time = time.time()
    samples_clamped = ppf_clamped(probabilities, **params)
    clamped_time = time.time()
    samples_interpolated = ppf_interpolated(probabilities, **params)
    interp_time = time.time()

    # For log_space, handle array input properly
    samples_log_space = np.array([ppf_log_space(p, **params) for p in probabilities])

    # Store all samples
    all_samples = {
        "Original": samples_original,
        "Scaled": samples_scaled,
        "High Precision": samples_high_precision,
        "Log Space": samples_log_space,
        "Clamped": samples_clamped,
        "Interpolated": samples_interpolated,
    }

    return (
        all_samples,
        clamped_time,
        hp_time,
        interp_time,
        n_samples,
        orig_time,
        params,
        scaled_time,
        start_time,
    )


@app.cell
def _(all_samples, n_samples, np, params, plt):
    """Visualize sample arrays with matplotlib"""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"PPF Alternative Implementations - {n_samples:,} Samples\nParams: {params}",
        fontsize=16,
    )

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Plot histogram for each implementation
    for i, (method, samples) in enumerate(all_samples.items()):
        ax = axes_flat[i]

        # Plot histogram
        ax.hist(samples, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(
            f"{method}\nMean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}"
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Add vertical lines for parameters
        ax.axvline(params["mini"], color="red", linestyle="--", alpha=0.7, label="Min")
        ax.axvline(
            params["mode"], color="green", linestyle="--", alpha=0.7, label="Mode"
        )
        ax.axvline(params["maxi"], color="blue", linestyle="--", alpha=0.7, label="Max")

        if i == 0:  # Only add legend to first subplot
            ax.legend()

    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(clamped_time, hp_time, interp_time, orig_time, scaled_time, start_time):
    print(f"Orig time: {orig_time - start_time}")
    print(f"Scaled time: {scaled_time - orig_time}")
    print(f"HP time: {hp_time - scaled_time}")
    print(f"Clamp time: {clamped_time - hp_time}")
    print(f"Interp time: {interp_time - clamped_time}")

    return


@app.cell
def _(all_samples, np):
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    print(f"{'Method':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)

    for _method, _samples in all_samples.items():
        print(
            f"{_method:<15} {np.mean(_samples):<10.4f} {np.std(_samples):<10.4f} {np.min(_samples):<10.4f} {np.max(_samples):<10.4f}"
        )
    return


@app.cell
def _(all_samples, np):
    for _method, _samples in all_samples.items():
        print(
            f"{_method:<15} {np.count_nonzero(np.isnan(_samples))}"
        )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
