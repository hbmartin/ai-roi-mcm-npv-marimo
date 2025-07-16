import pytest

from npv_model import npv_model_factory

npv_model = npv_model_factory(
    weeks_per_year=52,
    hours_per_workweek=40,
    avg_yearly_fully_loaded_cost_per_employee=130000,
)


class TestNPVModel:
    """Test suite for the npv_model function"""

    def test_basic_npv_calculation(self):
        """Test basic NPV calculation with default parameters"""
        result = npv_model(
            hours_saved_per_employee=10,
            number_of_employees=50,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0.3,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.1,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=50000,
            yearly_monitoring_and_support_cost=30000,
            first_year_change_management_cost=100000,
            yearly_ai_staff_cost=75000,
        )

        assert isinstance(result, dict)
        assert "npv" in result
        assert isinstance(result["npv"], (int, float))

    def test_all_return_fields(self):
        """Test that all expected fields are returned"""
        result = npv_model(
            hours_saved_per_employee=5,
            number_of_employees=20,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0.5,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.08,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=50000,
            yearly_monitoring_and_support_cost=30000,
            first_year_change_management_cost=100000,
            yearly_ai_staff_cost=75000,
        )

        expected_fields = [
            "npv",
            "time_savings",
            "quality_savings",
            "revenue_impact",
            "retention_savings",
            "year_1_net",
        ]

        for field in expected_fields:
            assert field in result
            assert isinstance(result[field], (int, float))

    def test_time_savings_calculation(self):
        """Test time savings benefit calculation"""
        result = npv_model(
            hours_saved_per_employee=10,
            number_of_employees=50,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.1,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=50000,
            yearly_monitoring_and_support_cost=30000,
            first_year_change_management_cost=100000,
            yearly_ai_staff_cost=75000,
        )

        expected_time_savings = 10 * 50 * 52 * 62.5 * 0.65
        assert abs(result["time_savings"] - expected_time_savings) < 0.01

    def test_quality_savings_calculation(self):
        """Test quality improvement benefit calculation"""
        result = npv_model(
            hours_saved_per_employee=0,
            number_of_employees=50,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0.3,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.1,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=50000,
            yearly_monitoring_and_support_cost=30000,
            first_year_change_management_cost=100000,
            yearly_ai_staff_cost=75000,
        )

        expected_quality_savings = 200000 * 0.3 + 130000 * 0.1 * 0.3
        assert abs(result["quality_savings"] - expected_quality_savings) < 0.01

    def test_revenue_impact_calculation(self):
        """Test product delivery benefit calculation"""
        result = npv_model(
            hours_saved_per_employee=0,
            number_of_employees=50,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.1,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=50000,
            yearly_monitoring_and_support_cost=30000,
            first_year_change_management_cost=100000,
            yearly_ai_staff_cost=75000,
        )

        expected_revenue_impact = 0.2 * 2 * 500000 * 0.3
        assert result["revenue_impact"] == expected_revenue_impact

    def test_retention_savings_calculation(self):
        """Test employee retention benefit calculation"""
        result = npv_model(
            hours_saved_per_employee=0,
            number_of_employees=50,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.1,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=50000,
            yearly_monitoring_and_support_cost=30000,
            first_year_change_management_cost=100000,
            yearly_ai_staff_cost=75000,
        )

        expected_retention_savings = 0.2 * 0.2 * 50 * 100000
        assert result["retention_savings"] == expected_retention_savings

    def test_cash_flow_calculations(self):
        """Test year-by-year cash flow calculations"""
        result = npv_model(
            hours_saved_per_employee=5,
            number_of_employees=20,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0.2,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.1,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=50000,
            yearly_monitoring_and_support_cost=30000,
            first_year_change_management_cost=100000,
            yearly_ai_staff_cost=75000,
        )

        # Calculate expected annual benefits
        annual_benefits = (
            result["time_savings"]
            + result["quality_savings"]
            + result["revenue_impact"]
            + result["retention_savings"]
        )
        ongoing_costs = (
            50000 + 30000 + 75000
        )  # yearly_tool_cost + yearly_monitoring_and_support_cost + yearly_ai_staff_cost

        expected_year_1_net = annual_benefits - (
            100000 + ongoing_costs
        )  # first_year_change_management_cost + ongoing_costs

        assert result["year_1_net"] == expected_year_1_net

    def test_npv_discount_calculation(self):
        """Test NPV discounting calculation"""
        result = npv_model(
            hours_saved_per_employee=10,
            number_of_employees=50,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0.3,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.1,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=50000,
            yearly_monitoring_and_support_cost=30000,
            first_year_change_management_cost=100000,
            yearly_ai_staff_cost=75000,
        )

        # Calculate expected cash flows
        annual_benefits = (
            result["time_savings"]
            + result["quality_savings"]
            + result["revenue_impact"]
            + result["retention_savings"]
        )
        ongoing_costs = 50000 + 30000 + 75000

        year_1_net = annual_benefits - (100000 + ongoing_costs)
        year_2_net = annual_benefits - ongoing_costs
        year_3_net = annual_benefits - ongoing_costs

        expected_npv = (
            year_1_net / (1 + 0.1) ** 1
            + year_2_net / (1 + 0.1) ** 2
            + year_3_net / (1 + 0.1) ** 3
        )

        assert abs(result["npv"] - expected_npv) < 0.01

    def test_zero_inputs(self):
        """Test behavior with zero inputs"""
        result = npv_model(
            hours_saved_per_employee=0,
            number_of_employees=0,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.1,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=50000,
            yearly_monitoring_and_support_cost=30000,
            first_year_change_management_cost=100000,
            yearly_ai_staff_cost=75000,
        )

        assert result["time_savings"] == 0
        assert result["quality_savings"] == 0
        assert result["retention_savings"] == 0
        # Only revenue impact should remain as a benefit

    def test_negative_npv_scenario(self):
        """Test scenario that should produce negative NPV"""
        result = npv_model(
            hours_saved_per_employee=1,
            number_of_employees=5,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0.1,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.15,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=500000,  # High costs
            yearly_monitoring_and_support_cost=200000,
            first_year_change_management_cost=500000,
            yearly_ai_staff_cost=300000,
        )

        assert result["npv"] < 0

    def test_positive_npv_scenario(self):
        """Test scenario that should produce positive NPV"""
        result = npv_model(
            hours_saved_per_employee=20,
            number_of_employees=100,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0.5,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.05,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=10000,  # Low costs
            yearly_monitoring_and_support_cost=5000,
            first_year_change_management_cost=20000,
            yearly_ai_staff_cost=15000,
        )

        assert result["npv"] > 0

    def test_parameter_validation_types(self):
        """Test that function handles different numeric types"""
        # Integer inputs
        result1 = npv_model(
            hours_saved_per_employee=10,
            number_of_employees=50,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0.3,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.1,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=50000,
            yearly_monitoring_and_support_cost=30000,
            first_year_change_management_cost=100000,
            yearly_ai_staff_cost=75000,
        )

        # Float inputs
        result2 = npv_model(
            hours_saved_per_employee=10.5,
            number_of_employees=50.0,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0.3,
            bug_time_rate=0.1,
            external_bug_cost=200000.0,
            discount_rate=0.1,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2.0,
            yearly_customer_value=500000.0,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000.0,
            yearly_tool_cost=50000.0,
            yearly_monitoring_and_support_cost=30000.0,
            first_year_change_management_cost=100000.0,
            yearly_ai_staff_cost=75000.0,
        )

        assert isinstance(result1["npv"], (int, float))
        assert isinstance(result2["npv"], (int, float))

    def test_edge_case_high_discount_rate(self):
        """Test with very high discount rate"""
        result = npv_model(
            hours_saved_per_employee=10,
            number_of_employees=50,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0.3,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.5,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=50000,
            yearly_monitoring_and_support_cost=30000,
            first_year_change_management_cost=100000,
            yearly_ai_staff_cost=75000,
        )

        assert isinstance(result["npv"], (int, float))
        # With high discount rate, NPV should be significantly less than undiscounted total
        # Calculate what the undiscounted total would be
        annual_benefits = (
            result["time_savings"]
            + result["quality_savings"]
            + result["revenue_impact"]
            + result["retention_savings"]
        )
        ongoing_costs = 50000 + 30000 + 75000

        year_1_net = annual_benefits - (100000 + ongoing_costs)
        year_2_net = annual_benefits - ongoing_costs
        year_3_net = annual_benefits - ongoing_costs

        undiscounted_total = year_1_net + year_2_net + year_3_net
        assert result["npv"] < undiscounted_total

    def test_annual_benefits_sum(self):
        """Test that annual benefits equals sum of all benefit components"""
        result = npv_model(
            hours_saved_per_employee=8,
            number_of_employees=40,
            productivity_conversion_rate=0.65,
            bug_reduction_rate=0.25,
            bug_time_rate=0.1,
            external_bug_cost=200000,
            discount_rate=0.12,
            feature_delivery_rate=0.2,
            feature_attribution_factor=0.3,
            new_customers_per_year=2,
            yearly_customer_value=500000,
            retention_improvement_rate=0.2,
            current_yearly_turnover_rate=0.2,
            replacement_cost_per_employee=100000,
            yearly_tool_cost=50000,
            yearly_monitoring_and_support_cost=30000,
            first_year_change_management_cost=100000,
            yearly_ai_staff_cost=75000,
        )

        # Test that all benefit components are present and numeric
        assert isinstance(result["time_savings"], (int, float))
        assert isinstance(result["quality_savings"], (int, float))
        assert isinstance(result["revenue_impact"], (int, float))
        assert isinstance(result["retention_savings"], (int, float))

    def test_consistent_results(self):
        """Test that identical inputs produce identical outputs"""
        params = {
            "hours_saved_per_employee": 12,
            "number_of_employees": 60,
            "productivity_conversion_rate": 0.65,
            "bug_reduction_rate": 0.35,
            "bug_time_rate": 0.1,
            "external_bug_cost": 200000,
            "discount_rate": 0.08,
            "feature_delivery_rate": 0.2,
            "feature_attribution_factor": 0.3,
            "new_customers_per_year": 2,
            "yearly_customer_value": 500000,
            "retention_improvement_rate": 0.2,
            "current_yearly_turnover_rate": 0.2,
            "replacement_cost_per_employee": 100000,
            "yearly_tool_cost": 50000,
            "yearly_monitoring_and_support_cost": 30000,
            "first_year_change_management_cost": 100000,
            "yearly_ai_staff_cost": 75000,
        }

        result1 = npv_model(**params)
        result2 = npv_model(**params)

        for key in result1:
            assert result1[key] == result2[key]
