# tests/core/test_cost_calculator.py
import unittest
from unittest.mock import patch
from extrai.core.cost_calculator import calculate_cost, ModelCosts


class TestCostCalculator(unittest.TestCase):
    def setUp(self):
        self.mock_costs = {
            "gpt-4-turbo": ModelCosts(
                input_cost_per_million=10.0, output_cost_per_million=30.0
            )
        }

    def test_calculate_cost_known_model(self):
        # Test with a known model
        with patch("extrai.core.cost_calculator.MODEL_COSTS", self.mock_costs):
            cost = calculate_cost("gpt-4-turbo", 1000, 2000)
            self.assertAlmostEqual(cost, 0.07)

    def test_calculate_cost_unknown_model(self):
        # Test with an unknown model
        with patch("extrai.core.cost_calculator.MODEL_COSTS", self.mock_costs):
            cost = calculate_cost("unknown-model", 1000, 2000)
            self.assertIsNone(cost)

    def test_calculate_cost_zero_tokens(self):
        # Test with zero tokens
        with patch("extrai.core.cost_calculator.MODEL_COSTS", self.mock_costs):
            cost = calculate_cost("gpt-4-turbo", 0, 0)
            self.assertEqual(cost, 0)


if __name__ == "__main__":
    unittest.main()
