import pytest

from npv_model import npv_model


class TestNPVModel:
    """Test suite for the npv_model function"""

    def test_basic_npv_calculation(self):
        """Test basic NPV calculation with default parameters"""
        result = npv_model(
            hours_saved=10,
            employees=50,
            hourly_rate=50,
            bug_reduction=0.3,
            discount_rate=0.1,
        )

        assert isinstance(result, dict)
        assert "npv" in result
        assert "annual_benefits" in result
        assert isinstance(result["npv"], (int, float))
        assert isinstance(result["annual_benefits"], (int, float))

    def test_all_return_fields(self):
        """Test that all expected fields are returned"""
        result = npv_model(
            hours_saved=5,
            employees=20,
            hourly_rate=40,
            bug_reduction=0.5,
            discount_rate=0.08,
        )

        expected_fields = [
            "npv",
            "annual_benefits",
            "time_savings",
            "quality_savings",
            "revenue_impact",
            "retention_savings",
            "year_1_net",
            "year_2_net",
            "year_3_net",
        ]

        for field in expected_fields:
            assert field in result
            assert isinstance(result[field], (int, float))

    def test_time_savings_calculation(self):
        """Test time savings benefit calculation"""
        result = npv_model(
            hours_saved=10,
            employees=50,
            hourly_rate=50,
            bug_reduction=0,
            discount_rate=0.1,
            productivity_conversion=0.65,
            weeks_per_year=52,
        )

        expected_time_savings = 10 * 50 * 52 * 50 * 0.65
        assert result["time_savings"] == expected_time_savings

    def test_quality_savings_calculation(self):
        """Test quality improvement benefit calculation"""
        result = npv_model(
            hours_saved=0,
            employees=50,
            hourly_rate=50,
            bug_reduction=0.3,
            discount_rate=0.1,
            current_bug_cost=200000,
            cost_factor=3.5,
        )

        expected_quality_savings = 0.3 * 200000 * 3.5
        assert result["quality_savings"] == expected_quality_savings

    def test_revenue_impact_calculation(self):
        """Test product delivery benefit calculation"""
        result = npv_model(
            hours_saved=0,
            employees=50,
            hourly_rate=50,
            bug_reduction=0,
            discount_rate=0.1,
            delivery_improvement=0.2,
            additional_customers=2,
            customer_value=500000,
            attribution_factor=0.3,
        )

        expected_revenue_impact = 0.2 * 2 * 500000 * 0.3
        assert result["revenue_impact"] == expected_revenue_impact

    def test_retention_savings_calculation(self):
        """Test employee retention benefit calculation"""
        result = npv_model(
            hours_saved=0,
            employees=50,
            hourly_rate=50,
            bug_reduction=0,
            discount_rate=0.1,
            retention_improvement=0.2,
            current_turnover=0.2,
            replacement_cost=100000,
        )

        expected_retention_savings = 0.2 * 0.2 * 50 * 100000
        assert result["retention_savings"] == expected_retention_savings

    def test_cash_flow_calculations(self):
        """Test year-by-year cash flow calculations"""
        result = npv_model(
            hours_saved=5,
            employees=20,
            hourly_rate=40,
            bug_reduction=0.2,
            discount_rate=0.1,
            impl_costs=550000,
            ongoing_costs=155000,
        )

        annual_benefits = result["annual_benefits"]

        assert result["year_1_net"] == annual_benefits - 550000
        assert result["year_2_net"] == annual_benefits - 155000
        assert result["year_3_net"] == annual_benefits - 155000

    def test_npv_discount_calculation(self):
        """Test NPV discounting calculation"""
        result = npv_model(
            hours_saved=10,
            employees=50,
            hourly_rate=50,
            bug_reduction=0.3,
            discount_rate=0.1,
        )

        year_1_net = result["year_1_net"]
        year_2_net = result["year_2_net"]
        year_3_net = result["year_3_net"]

        expected_npv = (
            year_1_net / (1 + 0.1) ** 1
            + year_2_net / (1 + 0.1) ** 2
            + year_3_net / (1 + 0.1) ** 3
        )

        assert abs(result["npv"] - expected_npv) < 0.01

    def test_zero_inputs(self):
        """Test behavior with zero inputs"""
        result = npv_model(
            hours_saved=0,
            employees=0,
            hourly_rate=0,
            bug_reduction=0,
            discount_rate=0.1,
        )

        assert result["time_savings"] == 0
        assert result["quality_savings"] == 0
        assert result["retention_savings"] == 0
        assert result["annual_benefits"] == result["revenue_impact"]

    def test_negative_npv_scenario(self):
        """Test scenario that should produce negative NPV"""
        result = npv_model(
            hours_saved=1,
            employees=5,
            hourly_rate=20,
            bug_reduction=0.1,
            discount_rate=0.15,
            impl_costs=1000000,
            ongoing_costs=500000,
        )

        assert result["npv"] < 0

    def test_positive_npv_scenario(self):
        """Test scenario that should produce positive NPV"""
        result = npv_model(
            hours_saved=20,
            employees=100,
            hourly_rate=75,
            bug_reduction=0.5,
            discount_rate=0.05,
            impl_costs=100000,
            ongoing_costs=50000,
        )

        assert result["npv"] > 0

    def test_parameter_validation_types(self):
        """Test that function handles different numeric types"""
        # Integer inputs
        result1 = npv_model(
            hours_saved=10,
            employees=50,
            hourly_rate=50,
            bug_reduction=0.3,
            discount_rate=0.1,
        )

        # Float inputs
        result2 = npv_model(
            hours_saved=10.5,
            employees=50.0,
            hourly_rate=50.75,
            bug_reduction=0.3,
            discount_rate=0.1,
        )

        assert isinstance(result1["npv"], (int, float))
        assert isinstance(result2["npv"], (int, float))

    def test_edge_case_high_discount_rate(self):
        """Test with very high discount rate"""
        result = npv_model(
            hours_saved=10,
            employees=50,
            hourly_rate=50,
            bug_reduction=0.3,
            discount_rate=0.5,
        )

        assert isinstance(result["npv"], (int, float))
        # With high discount rate, NPV should be significantly less than undiscounted total
        undiscounted_total = (
            result["year_1_net"] + result["year_2_net"] + result["year_3_net"]
        )
        assert result["npv"] < undiscounted_total

    def test_annual_benefits_sum(self):
        """Test that annual benefits equals sum of all benefit components"""
        result = npv_model(
            hours_saved=8,
            employees=40,
            hourly_rate=60,
            bug_reduction=0.25,
            discount_rate=0.12,
        )

        calculated_total = (
            result["time_savings"]
            + result["quality_savings"]
            + result["revenue_impact"]
            + result["retention_savings"]
        )

        assert abs(result["annual_benefits"] - calculated_total) < 0.01

    def test_default_parameters(self):
        """Test that default parameters are used correctly"""
        result = npv_model(
            hours_saved=5,
            employees=25,
            hourly_rate=45,
            bug_reduction=0.4,
            discount_rate=0.09,
        )

        # Check that default values are being used in calculations
        assert result["time_savings"] == 5 * 25 * 52 * 45 * 0.65
        assert result["quality_savings"] == 0.4 * 200000 * 3.5
        assert result["revenue_impact"] == 0.2 * 2 * 500000 * 0.3
        assert result["retention_savings"] == 0.2 * 0.2 * 25 * 100000

    def test_consistent_results(self):
        """Test that identical inputs produce identical outputs"""
        params = {
            "hours_saved": 12,
            "employees": 60,
            "hourly_rate": 55,
            "bug_reduction": 0.35,
            "discount_rate": 0.08,
        }

        result1 = npv_model(**params)
        result2 = npv_model(**params)

        for key in result1:
            assert result1[key] == result2[key]
