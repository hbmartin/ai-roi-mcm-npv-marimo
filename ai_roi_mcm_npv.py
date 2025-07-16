# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "beta-pert-dist-scipy==0.1.6",
#     "marimo",
#     "marimo-scipy-utils==0.1.2",
#     "numpy==2.3.1",
# ]
# [tool.uv.sources]
# npv-model = { path = ".", editable = true }
# ///

import marimo

__generated_with = "0.14.11"
app = marimo.App()


@app.cell
async def _():
    import micropip
    await micropip.install("monaco-dict-utils", deps=False)
    return


@app.cell
def setup_1():
    import marimo as mo

    from matplotlib import pyplot as plt
    import numpy as np
    from betapert import pert
    from marimo_scipy_utils import (
        abbrev_format,
        display_sliders,
        generate_ranges,
        params_sliders,
    )
    from matplotlib import ticker
    from monaco_dict_utils import outvals_to_dict, sim_factory

    # from npv_model import npv_model_factory
    # from params_pert import pert_descriptions, pert_ranges, same_pert_ranges

    return (
        abbrev_format,
        display_sliders,
        generate_ranges,
        mo,
        np,
        outvals_to_dict,
        params_sliders,
        pert,
        plt,
        sim_factory,
        ticker,
    )


@app.cell
def _():
    pert_descriptions = {
        "mini": "Min",
        "mode": "Mode",
        "maxi": "Max",
    }


    def same_pert_ranges(  # noqa: PLR0913
        mini: float,
        mini_value: float,
        value: float,
        maxi_value: float,
        maxi: float,
        step: float,
    ) -> dict[str, dict[str, float | int]]:
        return {
            "mini": {
                "lower": mini,
                "upper": maxi,
                "value": mini_value,
                "step": step,
            },
            "mode": {
                "lower": mini,
                "upper": maxi,
                "value": value,
                "step": step,
            },
            "maxi": {
                "lower": mini,
                "upper": maxi,
                "value": maxi_value,
                "step": step,
            },
        }


    def pert_ranges(  # noqa: PLR0913
        mini_min: float,
        mini_value: float,
        mini_max: float,
        mode_min: float,
        mode_value: float,
        mode_max: float,
        maxi_min: float,
        maxi_value: float,
        maxi_max: float,
        step: float,
    ) -> dict[str, dict[str, float | int]]:
        return {
            "mini": {
                "lower": mini_min,
                "upper": mini_max,
                "value": mini_value,
                "step": step,
            },
            "mode": {
                "lower": mode_min,
                "upper": mode_max,
                "value": mode_value,
                "step": step,
            },
            "maxi": {
                "lower": maxi_min,
                "upper": maxi_max,
                "value": maxi_value,
                "step": step,
            },
        }


    def npv_model_factory(
        weeks_per_year: float,
        hours_per_workweek: float,
        avg_yearly_fully_loaded_cost_per_employee: float,
    ):
        hours_per_year = weeks_per_year * hours_per_workweek
        hourly_rate = avg_yearly_fully_loaded_cost_per_employee / hours_per_year

        def npv_model(  # noqa: PLR0913
            hours_saved_per_employee: float,
            number_of_employees: float,
            productivity_conversion_rate: float,
            bug_reduction_rate: float,
            bug_time_rate: float,
            external_bug_cost: float,
            discount_rate: float,
            feature_delivery_rate: float,
            feature_attribution_factor: float,
            new_customers_per_year: float,
            yearly_customer_value: float,
            retention_improvement_rate: float,
            current_yearly_turnover_rate: float,
            replacement_cost_per_employee: float,
            yearly_tool_cost: float,
            yearly_monitoring_and_support_cost: float,
            first_year_change_management_cost: float,
            yearly_ai_staff_cost: float,
        ) -> dict:
            # Benefits Calculations
            # 1. Time Savings Benefits

            annual_time_savings = (
                hours_saved_per_employee
                * number_of_employees
                * weeks_per_year
                * hourly_rate
                * productivity_conversion_rate
            )

            # 2. Quality Improvement Benefits
            external_bug_saving = external_bug_cost * bug_reduction_rate
            internal_bug_saving = (
                avg_yearly_fully_loaded_cost_per_employee
                * bug_time_rate
                * bug_reduction_rate
            )
            annual_quality_savings = external_bug_saving + internal_bug_saving

            # 3. Product Delivery Benefits
            annual_revenue_impact = (
                feature_delivery_rate
                * new_customers_per_year
                * yearly_customer_value
                * feature_attribution_factor
            )

            # 4. Employee Retention Benefits
            annual_retention_savings = (
                retention_improvement_rate
                * current_yearly_turnover_rate
                * number_of_employees
                * replacement_cost_per_employee
            )

            # Total Annual Benefits
            total_annual_benefits = (
                annual_time_savings
                + annual_quality_savings
                + annual_revenue_impact
                + annual_retention_savings
            )

            ongoing_costs = (
                yearly_tool_cost + yearly_monitoring_and_support_cost + yearly_ai_staff_cost
            )

            # Cash Flows
            year_1_net_flow = total_annual_benefits - (
                first_year_change_management_cost + ongoing_costs
            )
            year_2_net_flow = total_annual_benefits - ongoing_costs
            year_3_net_flow = total_annual_benefits - ongoing_costs

            # NPV Calculation
            npv = (
                year_1_net_flow / (1 + discount_rate) ** 1
                + year_2_net_flow / (1 + discount_rate) ** 2
                + year_3_net_flow / (1 + discount_rate) ** 3
            )

            return {
                "npv": npv,
                "time_savings": annual_time_savings,
                "quality_savings": annual_quality_savings,
                "revenue_impact": annual_revenue_impact,
                "retention_savings": annual_retention_savings,
                "year_1_net": year_1_net_flow,
            }

        return npv_model
    return npv_model_factory, pert_descriptions, pert_ranges, same_pert_ranges


@app.cell
def _(mo):
    mo.md(
        r"""
    # AI ROI Monte Carlo NPV Analysis

    ### Assumptions:

    - Working weeks per year is set at 48 to account for holidays, vacation, and illness.
    - Work week is set at 40 hours
    """,
    )
    return


@app.cell
def _():
    invars = {}
    factory_vars = {
        "weeks_per_year": 48,
        "hours_per_workweek": 40,
    }
    return factory_vars, invars


@app.cell
def _(generate_ranges):
    my_params = generate_ranges(
        "triang",
        {"loc": {"lower": 0, "upper": 10}, "scale": {"upper": 5}},
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 1. Time Savings Benefits (Internal Productivity / Doing More)""")
    return


@app.cell
def _(params_sliders, pert_ranges):
    hspe = params_sliders(pert_ranges(0, 2, 5, 0, 5, 8, 2, 8, 12, step=0.25))
    return (hspe,)


@app.cell
def _(display_sliders, hspe, invars, pert, pert_descriptions):
    # display_params must be called in a seperate cell than the where sliders were created
    display_sliders(
        name="Hours Saved per Employee",
        sliders=hspe,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(mo):
    no_employees = mo.ui.slider(
        start=30,
        stop=100,
        step=1,
        value=50,
    )
    return (no_employees,)


@app.cell
def _(display_sliders, invars, no_employees):
    display_sliders("Number of Employees", no_employees, invars)
    return


@app.cell
def _(mo):
    aycpe = mo.ui.slider(
        start=80000,
        stop=380000,
        step=5000,
        value=230000,
    )
    return (aycpe,)


@app.cell
def _(aycpe, display_sliders, factory_vars):
    display_sliders(
        "Avg Yearly Fully Loaded Cost Per Employee ($)",
        aycpe,
        factory_vars,
    )
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    prodconv = params_sliders(same_pert_ranges(0, 0.3, 0.5, 0.7, 1, step=0.05))
    return (prodconv,)


@app.cell
def _(display_sliders, invars, pert, pert_descriptions, prodconv):
    display_sliders(
        name="Productivity Conversion Rate",
        sliders=prodconv,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""Productivity conversion is the rate at which saved time is re-deployed as work""",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 2. Quality Improvement Benefits (Preventing post-release bugs)""")
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    bug_reduction = params_sliders(same_pert_ranges(-1, 0, 0.2, 0.3, 1, step=0.05))
    return (bug_reduction,)


@app.cell
def _(bug_reduction, display_sliders, invars, pert, pert_descriptions):
    display_sliders(
        name="Bug Reduction Rate",
        sliders=bug_reduction,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    bug_time = params_sliders(same_pert_ranges(0, 0.2, 0.3, 0.6, 1, step=0.05))
    return (bug_time,)


@app.cell
def _(bug_time, display_sliders, invars, pert, pert_descriptions):
    display_sliders(
        name="Bug Time Rate",
        sliders=bug_time,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""Bug time rate is the fraction of employee time lost to remediating issues including internal issue management, process disruption, staff burnout, and customer support.""",
    )
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    ext_bug_cost = params_sliders(
        same_pert_ranges(0, 0, 20000, 100000, 1000000, step=10000),
    )
    return (ext_bug_cost,)


@app.cell
def _(display_sliders, ext_bug_cost, invars, pert, pert_descriptions):
    display_sliders(
        name="External Bug Cost ($)",
        sliders=ext_bug_cost,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    External bug cost (in dollars) represents the yearly value of customer non-renewal and new customer acquisitions lost due to production bugs (whether in the B2B interface or consumer apps).

    - This is currently set very low based on the expectation that customer renewal/acquisition is not substantially correlated to bugs. 
    - However, this also includes a risk adjusted cost of a catastraphic bug i.e. one that severly impacts ability to acquire new customers. (e.g. a bug costing $1m/yr at a 1%/yr risk = $10k)
    """,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 3. Product Delivery Benefits (External)""")
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    feat_rate = params_sliders(same_pert_ranges(0, 0.1, 0.25, 0.4, 1, step=0.05))
    return (feat_rate,)


@app.cell
def _(display_sliders, feat_rate, invars, pert, pert_descriptions):
    display_sliders(
        name="Feature Delivery Rate",
        sliders=feat_rate,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    feat_attr_rate = params_sliders(same_pert_ranges(0, 0, 0.1, 0.2, 1, step=0.05))
    return (feat_attr_rate,)


@app.cell
def _(display_sliders, feat_attr_rate, invars, pert, pert_descriptions):
    display_sliders(
        name="Feature Attribution Factor",
        sliders=feat_attr_rate,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The fraction of new customer acquisition or current customer retention attributed to new features or other developments.

    This is currently set very low based on the expectation that customer renewal/acquisition is not substantially correlated to new features.
    """,
    )
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    new_cust = params_sliders(same_pert_ranges(0, 1, 2, 3, 10, step=0.5))
    return (new_cust,)


@app.cell
def _(display_sliders, invars, new_cust, pert, pert_descriptions):
    display_sliders(
        name="New Customers per Year",
        sliders=new_cust,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    cust_value = params_sliders(
        same_pert_ranges(10000, 100000, 500000, 1000000, 5000000, step=5000),
    )
    return (cust_value,)


@app.cell
def _(cust_value, display_sliders, invars, pert, pert_descriptions):
    display_sliders(
        name="Yearly Customer Value",
        sliders=cust_value,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 3. Employee Retention Benefits""")
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    ret_imrpov_rate = params_sliders(same_pert_ranges(0, 0, 0.1, 0.4, 1, step=0.05))
    return (ret_imrpov_rate,)


@app.cell
def _(display_sliders, invars, pert, pert_descriptions, ret_imrpov_rate):
    display_sliders(
        name="Retention Improvement Rate",
        sliders=ret_imrpov_rate,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(mo):
    turnover_rate = mo.ui.slider(
        start=0,
        stop=1,
        step=0.05,
        value=0.2,
    )
    return (turnover_rate,)


@app.cell
def _(display_sliders, invars, turnover_rate):
    display_sliders("Current Yearly Turnover Rate", turnover_rate, invars)
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    replacement_cost = params_sliders(
        same_pert_ranges(20000, 60000, 75000, 90000, 120000, step=5000),
    )
    return (replacement_cost,)


@app.cell
def _(display_sliders, invars, pert, pert_descriptions, replacement_cost):
    display_sliders(
        name="Replacement Cost per Employee",
        sliders=replacement_cost,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""Replacement cost includes recruiting, interviewing, onboarding, and lost productivity.""",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## AI Implementation Costs""")
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    yts = params_sliders(
        same_pert_ranges(20000, 30000, 50000, 70000, 100000, step=5000),
    )
    return (yts,)


@app.cell
def _(display_sliders, invars, pert, pert_descriptions, yts):
    display_sliders(
        name="Yearly Tool Cost",
        sliders=yts,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    monitoring = params_sliders(
        same_pert_ranges(5000, 10000, 15000, 25000, 100000, step=5000),
    )
    return (monitoring,)


@app.cell
def _(display_sliders, invars, monitoring, pert, pert_descriptions):
    display_sliders(
        name="Yearly Monitoring and Support Cost",
        sliders=monitoring,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    tcm = params_sliders(
        same_pert_ranges(10000, 15000, 25000, 45000, 100000, step=5000),
    )
    return (tcm,)


@app.cell
def _(display_sliders, invars, pert, pert_descriptions, tcm):
    display_sliders(
        name="First Year Change Management Cost",
        sliders=tcm,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(mo):
    ai_staff = mo.ui.slider(
        start=200000,
        stop=600000,
        step=5000,
        value=300000,
    )
    return (ai_staff,)


@app.cell
def _(ai_staff, display_sliders, invars, pert, pert_descriptions):
    display_sliders(
        name="Yearly AI Staff Cost",
        sliders=ai_staff,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return


@app.cell
def _(params_sliders, same_pert_ranges):
    disc_rate = params_sliders(same_pert_ranges(0, 0.15, 0.25, 0.35, 1, step=0.05))
    return (disc_rate,)


@app.cell
def _(disc_rate, display_sliders, invars, pert, pert_descriptions):
    discount_slider_display = display_sliders(
        name="Discount Rate",
        sliders=disc_rate,
        invars=invars,
        dist=pert,
        descriptions=pert_descriptions,
    )
    return (discount_slider_display,)


@app.cell
def _(mo):
    get_message, set_message = mo.state(
        "Click button above to run with updated parameters",
    )
    return get_message, set_message


@app.cell
def _(mo, set_message):
    run_button = mo.ui.run_button(
        kind="success",
        label="Run Simulation",
        full_width=True,
        on_change=lambda value: (
            set_message("Starting simulation...") if value else None
        ),
    )
    return (run_button,)


@app.cell
def _(discount_slider_display, get_message, mo, run_button):
    mo.vstack(
        [
            mo.md("## Simulation"),
            discount_slider_display,
            run_button,
            mo.md(get_message()),
        ],
    )
    return


@app.cell
def _(
    factory_vars,
    invars,
    mo,
    npv_model_factory,
    outvals_to_dict,
    run_button,
    set_message,
    sim_factory,
):
    mo.stop(not run_button.value)
    set_message("Running simulation...")
    sim = sim_factory(
        name="ai_roi_npv_analysis",
        model_factory=npv_model_factory,
        factory_vars=factory_vars,
        invars=invars,
        ndraws=100000,
        debug=True,
    )
    with mo.redirect_stderr():
        sim.runSim()
    set_message(f"Finished simulation in {sim.runtime}")
    results = outvals_to_dict(sim)
    return (results,)


@app.cell
def _(abbrev_format, mo, np, plt, results, ticker):
    _fig, _ax = plt.subplots(figsize=(10, 6))

    # NPV Distribution
    _ax.hist(
        results["Year_1_Net"],
        bins=100,
        alpha=1,
        color="lightblue",
        edgecolor="lightblue",
    )
    _ax.axvline(
        np.mean(results["Year_1_Net"]),
        color="red",
        linestyle="--",
        label=f"Mean: ${np.mean(results["Year_1_Net"]):,.0f}",
    )
    _ax.axvline(0, color="black", linestyle="-", alpha=0.5, label="Break-even")
    _ax.set_title("First Year Net Benefits")
    _ax.set_xlabel("NPV ($)")
    _ax.set_ylabel("Frequency")
    _ax.xaxis.set_major_formatter(ticker.FuncFormatter(abbrev_format))
    _ax.legend()
    _ax.grid(visible=True, alpha=0.3)

    plt.tight_layout()

    _p5 = np.percentile(results["Year_1_Net"], 5)
    _p95 = np.percentile(results["Year_1_Net"], 95)
    _positive_pct = (np.array(results["Year_1_Net"]) > 0).mean() * 100

    _results_text = f"""
    **NPV Monte Carlo Results:**

    - 90% Confidence Interval: [${_p5:,.0f}, ${_p95:,.0f}]
    - Probability of Positive NPV: {_positive_pct:.1f}%
    - Number of Simulations: {len(results["Year_1_Net"]):,}
    """

    mo.vstack(
        [mo.md("### Results\n#### Year 1 ROI"), mo.as_html(_fig), mo.md(_results_text)],
    )
    return


@app.cell
def plot_results(abbrev_format, mo, np, plt, results, ticker):
    _fig, _ax = plt.subplots(figsize=(10, 6))

    # NPV Distribution
    _ax.hist(
        results["Npv"],
        bins=100,
        alpha=1,
        color="lightblue",
        edgecolor="lightblue",
    )
    _ax.axvline(
        np.mean(results["Npv"]),
        color="red",
        linestyle="--",
        label=f"Mean: ${np.mean(results["Npv"]):,.0f}",
    )
    _ax.axvline(0, color="black", linestyle="-", alpha=0.5, label="Break-even")
    _ax.set_title("3 Year ROI NPV Distribution")
    _ax.set_xlabel("NPV ($)")
    _ax.set_ylabel("Frequency")
    _ax.xaxis.set_major_formatter(ticker.FuncFormatter(abbrev_format))
    _ax.legend()
    _ax.grid(visible=True, alpha=0.3)

    plt.tight_layout()

    _p5 = np.percentile(results["Npv"], 5)
    _p95 = np.percentile(results["Npv"], 95)
    _positive_pct = (np.array(results["Npv"]) > 0).mean() * 100

    _results_text = f"""
    **NPV Monte Carlo Results:**

    - 90% Confidence Interval: [${_p5:,.0f}, ${_p95:,.0f}]
    - Probability of Positive NPV: {_positive_pct:.1f}%
    - Number of Simulations: {len(results["Npv"]):,}
    """

    mo.vstack([mo.md("### 3 Year NPV ROI"), mo.as_html(_fig), mo.md(_results_text)])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
