from typing import Callable


def npv_model_factory(
    weeks_per_year: float | int,
    hours_per_workweek: float | int,
    avg_yearly_fully_loaded_cost_per_employee: float | int,
) -> Callable[
    [
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    dict,
]:
    hours_per_year = weeks_per_year * hours_per_workweek
    hourly_rate = avg_yearly_fully_loaded_cost_per_employee / hours_per_year

    def npv_model(
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
            "annual_benefits": total_annual_benefits,
            "time_savings": annual_time_savings,
            "quality_savings": annual_quality_savings,
            "revenue_impact": annual_revenue_impact,
            "retention_savings": annual_retention_savings,
            "year_1_net": year_1_net_flow,
            "year_2_net": year_2_net_flow,
            "year_3_net": year_3_net_flow,
        }

    return npv_model
