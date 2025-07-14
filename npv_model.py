def npv_model(
    hours_saved,
    employees,
    hourly_rate,
    bug_reduction,
    discount_rate,
    productivity_rate=0.65,
    weeks_per_year=52,
    current_bug_cost=200000,
    cost_factor=3.5,
    delivery_improvement=0.2,
    additional_customers=2,
    customer_value=500000,
    attribution_factor=0.3,
    retention_improvement=0.2,
    current_turnover=0.2,
    replacement_cost=100000,
    impl_costs=550000,
    ongoing_costs=155000,
):
    """
    Calculate 3-year NPV for AI implementation
    """

    # Benefits Calculations
    # 1. Time Savings Benefits
    annual_time_savings = (
        hours_saved * employees * weeks_per_year * hourly_rate * productivity_rate
    )

    # 2. Quality Improvement Benefits
    annual_quality_savings = bug_reduction * current_bug_cost * cost_factor

    # 3. Product Delivery Benefits
    annual_revenue_impact = (
        delivery_improvement
        * additional_customers
        * customer_value
        * attribution_factor
    )

    # 4. Employee Retention Benefits
    annual_retention_savings = (
        retention_improvement * current_turnover * employees * replacement_cost
    )

    # Total Annual Benefits
    total_annual_benefits = (
        annual_time_savings
        + annual_quality_savings
        + annual_revenue_impact
        + annual_retention_savings
    )

    # Cash Flows
    year_1_net_flow = total_annual_benefits - impl_costs
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
        "annual_benefits": total_annual_benefits,
        "time_savings": annual_time_savings,
        "quality_savings": annual_quality_savings,
        "revenue_impact": annual_revenue_impact,
        "retention_savings": annual_retention_savings,
        "year_1_net": year_1_net_flow,
        "year_2_net": year_2_net_flow,
        "year_3_net": year_3_net_flow,
    }
