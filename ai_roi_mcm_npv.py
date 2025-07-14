import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell(hide_code=True)
def setup_1():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import monaco

    return mo, monaco, norm, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# AI ROI Monte Carlo NPV Analysis""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
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
    n_samples_slider = mo.ui.slider(
        start=1000, stop=10000, step=500, value=5000, label="Number of Simulations"
    )

    return (
        bug_reduction_slider,
        discount_rate_slider,
        employees_slider,
        hourly_rate_slider,
        hours_saved_slider,
        n_samples_slider,
    )


@app.cell
def _(
    bug_reduction_slider,
    discount_rate_slider,
    employees_slider,
    hourly_rate_slider,
    hours_saved_slider,
    mo,
    n_samples_slider,
):
    mo.hstack(
        [
            mo.vstack([hours_saved_slider, employees_slider, hourly_rate_slider]),
            mo.vstack([bug_reduction_slider, discount_rate_slider, n_samples_slider]),
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
    n_samples_slider,
):
    # Get current parameter values
    hours_saved = hours_saved_slider.value
    employees = employees_slider.value
    hourly_rate = hourly_rate_slider.value
    bug_reduction = bug_reduction_slider.value
    discount_rate = discount_rate_slider.value
    n_samples = int(n_samples_slider.value)

    mo.md(
        f"""**Current Parameters:** 
    Hours Saved: {hours_saved}/week, Employees: {employees}, 
    Hourly Rate: ${hourly_rate}, Bug Reduction: {bug_reduction:.0%}, 
    Discount Rate: {discount_rate:.0%}, Simulations: {n_samples}"""
    )
    return (
        bug_reduction,
        discount_rate,
        employees,
        hourly_rate,
        hours_saved,
        n_samples,
    )


@app.cell
def _(mo):
    mo.md(
        """
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
    mo.md(
        """
    ## Monte Carlo Simulation using Monaco

    Now let's use the monaco package to perform Monte Carlo simulation for NPV analysis.
    We'll model uncertainty in key parameters using appropriate probability distributions.
    """
    )
    return


@app.cell
def _(
    bug_reduction,
    discount_rate,
    employees,
    hourly_rate,
    hours_saved,
    monaco,
    n_samples,
    norm,
):
    from scipy.stats import uniform, triang, beta

    # Define simulation functions
    def preprocess(case):
        return (
            {
                "hours_saved": case.invals["hours_saved"].val,
                "employees": case.constvals["employees"],
                "hourly_rate": case.invals["hourly_rate"].val,
                "bug_reduction": case.invals["bug_reduction"].val,
                "discount_rate": case.invals["discount_rate"].val,
                "productivity_rate": case.invals["productivity_rate"].val,
                "current_bug_cost": case.invals["current_bug_cost"].val,
                "delivery_improvement": case.invals["delivery_improvement"].val,
                "retention_improvement": case.invals["retention_improvement"].val,
            },
        )

    def run(params):
        result = npv_model(**params)
        return (result,)

    def postprocess(case, simulation_output):
        case.addOutVal(name="NPV", val=simulation_output["npv"])
        case.addOutVal(name="Annual_Benefits", val=simulation_output["annual_benefits"])
        case.addOutVal(name="Time_Savings", val=simulation_output["time_savings"])
        case.addOutVal(name="Quality_Savings", val=simulation_output["quality_savings"])
        case.addOutVal(name="Year_1_Net", val=simulation_output["year_1_net"])

    # Create monaco simulation
    sim = monaco.Sim(
        name="ai_roi_npv_analysis",
        ndraws=n_samples,
        fcns={"preprocess": preprocess, "run": run, "postprocess": postprocess},
        debug=True,
        verbose=True,
    )

    ## Constants
    sim.addConstVal(name="employees", val=employees)

    ## Time Savings Benefits (Internal)
    sim.addInVar(
        name="hours_saved",
        dist=triang,
        distkwargs={"c": 0.5, "loc": hours_saved * 0.7, "scale": hours_saved * 0.6},
    )
    sim.addInVar(
        name="hourly_rate",
        dist=norm,
        distkwargs={"loc": hourly_rate, "scale": hourly_rate * 0.1},
    )
    sim.addInVar(
        name="productivity_rate", dist=uniform, distkwargs={"loc": 0.55, "scale": 0.2}
    )

    ## Quality Improvement Benefits
    # Bug reduction (beta distribution)
    sim.addInVar(
        name="bug_reduction",
        dist=beta,
        distkwargs={
            "a": 3,
            "b": 5,
            "loc": bug_reduction * 0.7,
            "scale": bug_reduction * 0.6,
        },
    )

    # Discount rate (uniform distribution)
    sim.addInVar(
        name="discount_rate",
        dist=uniform,
        distkwargs={"loc": discount_rate * 0.8, "scale": discount_rate * 0.4},
    )

    sim.addInVar(
        name="current_bug_cost",
        dist=triang,
        distkwargs={"c": 0.3, "loc": 150000, "scale": 100000},
    )

    ## Product Delivery Benefits
    sim.addInVar(
        name="delivery_improvement",
        dist=beta,
        distkwargs={"a": 2, "b": 3, "loc": 0.1, "scale": 0.2},
    )

    ## Employee Retention Benefits
    sim.addInVar(
        name="retention_improvement",
        dist=beta,
        distkwargs={"a": 2, "b": 4, "loc": 0.1, "scale": 0.25},
    )

    return (sim,)


@app.cell
def _(sim):
    sim.runSim()
    print(f"{sim.name} Runtime: {sim.runtime}")
    print(sim.outvars.keys())
    # Extract results
    npv_values = [case.outvals["NPV"].val for case in sim.cases]
    benefits_values = [case.outvals["Annual_Benefits"].val for case in sim.cases]
    time_savings_values = [case.outvals["Time_Savings"].val for case in sim.cases]
    quality_savings_values = [case.outvals["Quality_Savings"].val for case in sim.cases]
    print(npv_values)
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


@app.cell
def _(mo):
    mo.md(
        """
    ## Summary

    This AI ROI NPV analysis demonstrates:
    1. **Interactive parameter control** for key business variables
    2. **Comprehensive NPV modeling** including time savings, quality improvements, delivery acceleration, and retention benefits
    3. **Monte Carlo simulation** with realistic probability distributions for uncertainty modeling
    4. **Risk assessment** showing probability of positive NPV and confidence intervals
    5. **Visual analysis** of benefit components and their relationships
    6. **Sensitivity analysis** through parameter variation

    The model incorporates uncertainty in:
    - Hours saved per employee (triangular distribution)
    - Hourly rates and productivity factors (normal/uniform distributions)
    - Bug reduction effectiveness (beta distribution)
    - Business impact factors (various distributions)

    Adjust the sliders to see how different assumptions affect the NPV distribution and risk profile.
    """
    )
    return


if __name__ == "__main__":
    app.run()
