import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def setup_1():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import monaco
    from npv_model import npv_model
    from npv_sim import npv_sim
    from marimo_components import (
        distribution_params,
        display_params,
        generate_ranges,
    )
    from params_pert import pert_metadata, generate_pert_ranges
    from betapert import pert
    from params_to_sim import params_to_sim

    return (
        display_params,
        distribution_params,
        generate_pert_ranges,
        generate_ranges,
        mo,
        monaco,
        norm,
        np,
        npv_sim,
        params_to_sim,
        pert,
        pert_metadata,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""# AI ROI Monte Carlo NPV Analysis""")
    return


@app.cell
def _(generate_ranges):
    my_params = generate_ranges(
        "triang", {"loc": {"lower": 0, "upper": 10}, "scale": {"upper": 5}}
    )
    return


@app.cell
def _():
    invars = {}
    return (invars,)


@app.cell
def _(distribution_params, generate_pert_ranges):
    _pert_params = generate_pert_ranges(0, 2, 5, 0, 5, 8, 2, 8, 12)
    hspe = distribution_params(_pert_params)
    return (hspe,)


@app.cell
def _(display_params, hspe, invars, pert, pert_metadata):
    display_params("Hours Saved per Employee", hspe, pert, invars, pert_metadata)
    return


@app.cell
def _(invars):
    invars
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Interactive Parameters
    Adjust the key parameters for the AI ROI NPV Monte Carlo simulation:
    """
    )
    return


@app.cell
def _(mo):
    # Interactive widgets for NPV simulation parameters
    hours_saved_slider = mo.ui.slider(
        start=1, stop=10, step=0.5, value=4, label="Hours Saved per Employee per Week"
    )
    employees_slider = mo.ui.slider(
        start=20, stop=100, step=5, value=50, label="Number of Employees"
    )
    hourly_rate_slider = mo.ui.slider(
        start=50, stop=150, step=5, value=75, label="Fully-loaded Hourly Rate ($)"
    )
    bug_reduction_slider = mo.ui.slider(
        start=0.1, stop=0.6, step=0.05, value=0.35, label="Bug Reduction %"
    )
    discount_rate_slider = mo.ui.slider(
        start=0.08, stop=0.25, step=0.01, value=0.15, label="Discount Rate"
    )

    return (
        bug_reduction_slider,
        discount_rate_slider,
        employees_slider,
        hourly_rate_slider,
        hours_saved_slider,
    )


@app.cell
def _(
    bug_reduction_slider,
    discount_rate_slider,
    employees_slider,
    hourly_rate_slider,
    hours_saved_slider,
    mo,
):
    mo.hstack(
        [
            mo.vstack([hours_saved_slider, employees_slider, hourly_rate_slider]),
            mo.vstack([bug_reduction_slider, discount_rate_slider]),
        ]
    )
    return


@app.cell
def _(
    bug_reduction_slider,
    discount_rate_slider,
    employees_slider,
    hourly_rate_slider,
    hours_saved_slider,
    mo,
):
    # Get current parameter values
    hours_saved = hours_saved_slider.value
    employees = employees_slider.value
    hourly_rate = hourly_rate_slider.value
    bug_reduction = bug_reduction_slider.value
    discount_rate = discount_rate_slider.value
    # n_samples = int(n_samples_slider.value)

    mo.md(
        f"""**Current Parameters:** 
    Hours Saved: {hours_saved}/week, Employees: {employees}, 
    Hourly Rate: ${hourly_rate}, Bug Reduction: {bug_reduction:.0%}, 
    Discount Rate: {discount_rate:.0%}"""
    )
    return bug_reduction, discount_rate, employees, hourly_rate, hours_saved


@app.cell
def _(mo):
    mo.md(
        r"""
    ## NPV Calculation Function

    This function calculates the 3-year NPV for AI implementation based on:
    - Time savings benefits
    - Quality improvement benefits 
    - Product delivery improvements
    - Employee retention benefits
    - Implementation and ongoing costs
    """
    )
    return


@app.cell
def _(mo):
    n_samples_slider = mo.ui.slider(
        start=10000, stop=100000, step=1000, value=50000, label="Number of Simulations"
    ).form(submit_button_label="Run Simulation")
    n_samples_slider
    return (n_samples_slider,)


@app.cell
def _(n_samples_slider):
    n_samples = int(n_samples_slider.value) if n_samples_slider.value else None
    return (n_samples,)


@app.cell
def _(
    bug_reduction,
    discount_rate,
    employees,
    hourly_rate,
    hours_saved,
    invars,
    mo,
    monaco,
    n_samples,
    norm,
    npv_sim,
    params_to_sim,
):
    mo.stop(not n_samples)
    sim = npv_sim(
        bug_reduction,
        discount_rate,
        employees,
        hourly_rate,
        hours_saved,
        mo,
        monaco,
        n_samples,
        norm,
    )
    params_to_sim(sim, invars)
    sim.runSim()
    print(f"{sim.name} Runtime: {sim.runtime}")
    print(sim.outvars.keys())
    # Extract results
    npv_values = [case.outvals["NPV"].val for case in sim.cases]
    benefits_values = [case.outvals["Annual_Benefits"].val for case in sim.cases]
    time_savings_values = [case.outvals["Time_Savings"].val for case in sim.cases]
    quality_savings_values = [case.outvals["Quality_Savings"].val for case in sim.cases]
    return (
        benefits_values,
        npv_values,
        quality_savings_values,
        time_savings_values,
    )


@app.cell
def plot_results(
    benefits_values,
    mo,
    np,
    npv_values,
    plt,
    quality_savings_values,
    time_savings_values,
):
    # Create visualization of NPV results
    fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
    # NPV Distribution
    axes[0, 0].hist(
        npv_values, bins=50, alpha=0.7, color="lightblue", edgecolor="black"
    )
    axes[0, 0].axvline(
        np.mean(npv_values),
        color="red",
        linestyle="--",
        label=f"Mean: ${np.mean(npv_values):,.0f}",
    )
    axes[0, 0].axvline(0, color="black", linestyle="-", alpha=0.5, label="Break-even")
    axes[0, 0].set_title("NPV Distribution")
    axes[0, 0].set_xlabel("NPV ($)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Benefits Breakdown
    axes[0, 1].hist(
        benefits_values, bins=30, alpha=0.7, color="lightgreen", edgecolor="black"
    )
    axes[0, 1].set_title("Annual Benefits Distribution")
    axes[0, 1].set_xlabel("Annual Benefits ($)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # Time Savings vs Quality Savings
    scatter = axes[1, 0].scatter(
        time_savings_values,
        quality_savings_values,
        c=npv_values,
        cmap="RdYlGn",
        alpha=0.6,
    )
    axes[1, 0].set_xlabel("Time Savings ($)")
    axes[1, 0].set_ylabel("Quality Savings ($)")
    axes[1, 0].set_title("Time vs Quality Savings (colored by NPV)")
    plt.colorbar(scatter, ax=axes[1, 0])
    axes[1, 0].grid(True, alpha=0.3)

    # Risk Analysis
    positive_npv_pct = (np.array(npv_values) > 0).mean() * 100
    axes[1, 1].pie(
        [positive_npv_pct, 100 - positive_npv_pct],
        labels=[
            f"Positive NPV\n({positive_npv_pct:.1f}%)",
            f"Negative NPV\n({100-positive_npv_pct:.1f}%)",
        ],
        colors=["lightgreen", "lightcoral"],
        autopct="%1.1f%%",
    )
    axes[1, 1].set_title("NPV Risk Assessment")

    plt.tight_layout()

    # Calculate comprehensive statistics
    npv_mean = np.mean(npv_values)
    npv_std = np.std(npv_values)
    npv_p5 = np.percentile(npv_values, 5)
    npv_p95 = np.percentile(npv_values, 95)
    benefits_mean = np.mean(benefits_values)

    results_text = f"""
    **NPV Monte Carlo Results:**
    - Mean NPV: ${npv_mean:,.0f}
    - NPV Std Dev: ${npv_std:,.0f}
    - 90% Confidence Interval: [${npv_p5:,.0f}, ${npv_p95:,.0f}]
    - Probability of Positive NPV: {positive_npv_pct:.1f}%
    - Mean Annual Benefits: ${benefits_mean:,.0f}
    - Number of Simulations: {len(npv_values):,}
    """

    mo.vstack([mo.as_html(fig2), mo.md(results_text)])
    return


@app.cell
def _(mo, np, npv_values):
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = np.percentile(npv_values, percentiles)

    percentile_table = mo.ui.table(
        data=[
            {"Percentile": f"{p}%", "NPV": f"${v:,.0f}"}
            for p, v in zip(percentiles, percentile_values)
        ],
        selection=None,
    )

    mo.vstack([mo.md("**NPV Percentiles:**"), percentile_table])
    return


if __name__ == "__main__":
    app.run()
