import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import monaco
    return mo, monaco, norm, np, plt


@app.cell
def _(mo):
    mo.md("""# Monte Carlo Simulation with Normal Distribution""")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Interactive Parameters
    Adjust the parameters for the normal distribution and Monte Carlo simulation:
    """
    )
    return


@app.cell
def _(mo):
    # Interactive widgets for normal distribution parameters
    mean_slider = mo.ui.slider(
        start=-10, stop=10, step=0.1, value=0, label="Mean (μ)"
    )
    std_slider = mo.ui.slider(
        start=0.1, stop=5, step=0.1, value=1, label="Standard Deviation (σ)"
    )
    n_samples_slider = mo.ui.slider(
        start=100, stop=10000, step=100, value=1000, label="Number of Samples"
    )

    return mean_slider, n_samples_slider, std_slider


@app.cell
def _(mean_slider, mo, n_samples_slider, std_slider):
    mo.hstack([
        mo.vstack([mean_slider, std_slider]),
        n_samples_slider
    ])
    return


@app.cell
def _(mean_slider, mo, n_samples_slider, std_slider):
    # Get current parameter values
    mu = mean_slider.value
    sigma = std_slider.value
    n_samples = int(n_samples_slider.value)

    mo.md(f"**Current Parameters:** μ = {mu}, σ = {sigma}, n = {n_samples}")
    return mu, n_samples, sigma


@app.cell
def show_figure(mo, mu, n_samples, norm, np, plt, sigma):
    # Direct sampling from scipy.stats.norm
    samples_direct = norm.rvs(loc=mu, scale=sigma, size=n_samples)

    # Create histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of samples
    ax1.hist(samples_direct, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # Overlay theoretical PDF
    x = np.linspace(samples_direct.min(), samples_direct.max(), 100)
    theoretical_pdf = norm.pdf(x, loc=mu, scale=sigma)
    ax1.plot(x, theoretical_pdf, 'r-', linewidth=2, label='Theoretical PDF')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Direct Sampling from Normal Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot to check normality
    from scipy import stats
    stats.probplot(samples_direct, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Check)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Display statistics
    sample_mean = np.mean(samples_direct)
    sample_std = np.std(samples_direct, ddof=1)

    stats_text = f"""
    **Sample Statistics:**
    - Sample Mean: {sample_mean:.4f} (theoretical: {mu})
    - Sample Std: {sample_std:.4f} (theoretical: {sigma})
    - Sample Size: {len(samples_direct)}
    """

    mo.vstack([
        mo.as_html(fig),
        mo.md(stats_text)
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Monte Carlo Simulation using Monaco

    Now let's use the monaco package to perform a more structured Monte Carlo simulation.
    We'll create a simple model that uses normal random variables as inputs.
    """
    )
    return


@app.cell
def _(monaco, mu, n_samples, norm, sigma):
    # Define a simple model function that we want to analyze
    def example_model(x1, x2):
        """
        Example model: calculate some function of two normal random variables
        This could represent any complex model you want to analyze
        """
        return x1**2 + 2*x1*x2 + x2**2

    # Create monaco simulation
    sim = monaco.Sim(
        nCases=n_samples,
        fcnName='example_model'
    )

    # Add input variables with normal distributions
    sim.addInVar(name='x1', 
                 dist=lambda: norm.rvs(loc=mu, scale=sigma),
                 correlatesTo=[])

    sim.addInVar(name='x2', 
                 dist=lambda: norm.rvs(loc=mu, scale=sigma),
                 correlatesTo=[])

    # Add the model function
    sim.fcn = example_model

    return (sim,)


@app.cell
def _(mo, sim):
    # Run the Monte Carlo simulation
    try:
        sim.runSim()

        # Get results
        x1_values = [case.inVars['x1'].val for case in sim.cases]
        x2_values = [case.inVars['x2'].val for case in sim.cases]
        output_values = [case.outVal for case in sim.cases]

        simulation_success = True

    except Exception as e:
        mo.md(f"**Error running simulation:** {str(e)}")
        simulation_success = False
        x1_values, x2_values, output_values = [], [], []

    return output_values, simulation_success, x1_values, x2_values


@app.cell
def plot_results(
    mo,
    np,
    output_values,
    plt,
    simulation_success,
    x1_values,
    x2_values,
):
    if simulation_success:
        # Create visualization of results
        fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Input distributions
        axes[0,0].hist(x1_values, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0,0].set_title('Input Variable X1')
        axes[0,0].set_xlabel('X1 Value')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)

        axes[0,1].hist(x2_values, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].set_title('Input Variable X2')
        axes[0,1].set_xlabel('X2 Value')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True, alpha=0.3)

        # Output distribution
        axes[1,0].hist(output_values, bins=30, alpha=0.7, color='salmon', edgecolor='black')
        axes[1,0].set_title('Model Output Distribution')
        axes[1,0].set_xlabel('Output Value')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)

        # Scatter plot of inputs vs output
        scatter = axes[1,1].scatter(x1_values, x2_values, c=output_values, 
                                   cmap='viridis', alpha=0.6)
        axes[1,1].set_xlabel('X1')
        axes[1,1].set_ylabel('X2')
        axes[1,1].set_title('Input Space colored by Output')
        plt.colorbar(scatter, ax=axes[1,1])
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Calculate statistics
        output_mean = np.mean(output_values)
        output_std = np.std(output_values)
        output_min = np.min(output_values)
        output_max = np.max(output_values)

        results_text = f"""
        **Monte Carlo Results:**
        - Output Mean: {output_mean:.4f}
        - Output Std: {output_std:.4f}
        - Output Range: [{output_min:.4f}, {output_max:.4f}]
        - Number of Cases: {len(output_values)}
        """

        mo.vstack([
            mo.as_html(fig2),
            mo.md(results_text)
        ])
    else:
        mo.md("Simulation failed. Please check the parameters and try again.")
    return


@app.cell
def _(mo, np, output_values, simulation_success):
    if simulation_success:
        # Additional analysis
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = np.percentile(output_values, percentiles)

        percentile_table = mo.ui.table(
            data=[
                {"Percentile": f"{p}%", "Value": f"{v:.4f}"} 
                for p, v in zip(percentiles, percentile_values)
            ],
            selection=None
        )

        mo.vstack([
            mo.md("**Output Percentiles:**"),
            percentile_table
        ])
    else:
        mo.md("")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Summary

    This notebook demonstrates:
    1. **Interactive parameter control** using marimo widgets
    2. **Direct sampling** from scipy.stats.norm
    3. **Structured Monte Carlo simulation** using the monaco package
    4. **Comprehensive visualization** of inputs, outputs, and relationships
    5. **Statistical analysis** of simulation results

    You can adjust the mean, standard deviation, and number of samples to see how they affect both the direct sampling and the Monte Carlo simulation results.
    """
    )
    return


if __name__ == "__main__":
    app.run()
