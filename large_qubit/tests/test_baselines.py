"""Selection-only tests: no target evaluation or acquisition is performed."""

import copy
import unittest
from unittest.mock import patch

from large_qubit.baselines import nearest_training_geometry, retrieval_candidates, uniform_random
from large_qubit.tests._section5_fixtures import catalogue_fixture


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue = catalogue_fixture()
        cls.cost = {"release_sha256": "a" * 64, "logical_energy_requests": 6528, "wall_seconds": 1000.0}

    def row(self, row_id, energy=-1.0, *, alias="CO2_T01", ops=None, role="fitting"):
        return {"row_id": row_id, "geometry_id": alias, "role": role,
            "ops": [1] * 32 if ops is None else ops, "energy_hartree": energy,
            "catalogue_sha256": self.catalogue["sha256"]}

    def retrieve(self, rows, **kwargs):
        return retrieval_candidates("CO2_I01", rows, catalogue=self.catalogue,
            length=32, acquisition_cost=self.cost, **kwargs)

    def test_uniform_exact_pool_independent_slots_reproducibility(self):
        # Every slot invokes randrange(513); no identity removal or novelty filter.
        with patch("large_qubit.baselines.random.Random") as rng:
            rng.return_value.randrange.side_effect = [0, 512] * 32
            result = uniform_random(self.catalogue, length=32, count=2, seed=22)
            self.assertEqual(result["circuits"], [[0, 512] * 16] * 2)
            self.assertEqual(rng.return_value.randrange.call_count, 64)
            self.assertTrue(all(call.args == (513,) for call in rng.return_value.randrange.call_args_list))
        first = uniform_random(self.catalogue, length=64, count=3, seed=22)
        self.assertEqual(first, uniform_random(self.catalogue, length=64, count=3, seed=22))
        self.assertEqual(first["maximum_total_logical_requests"], 4)
        for count in (True, 1024, -1):
            with self.assertRaises(ValueError):
                uniform_random(self.catalogue, length=32, count=count, seed=22)

    def test_decimal_nearest_ties_lower_distance(self):
        self.assertEqual(nearest_training_geometry("CO2_I01"), "CO2_T01")  # exact midpoint1.10
        self.assertEqual(nearest_training_geometry("CO2_I02"), "CO2_T03")
        self.assertEqual(nearest_training_geometry("CO2_V01"), "CO2_T02")

    def test_fitting_order_tuple_then_row_id_and_full_cost(self):
        rows = [self.row("b", -2), self.row("z", -3), self.row("a", -2),
                self.row("c", -2, ops=[0] * 32),
                self.row("other", -1000, alias="CO2_T02"),
                self.row("reserved", -10000, role="diagnostic", ops="never inspect")]
        result = self.retrieve(rows)
        self.assertEqual([row["row_id"] for row in result["candidates"]], ["z", "c", "a", "b"])
        self.assertEqual(result["excluded_diagnostic_rows"], 1)
        self.assertEqual(result["excluded_other_training_geometry_rows"], 1)
        self.assertEqual(result["source_acquisition_cost"], self.cost)
        self.assertFalse(result["target_based_reordering"])
        self.assertFalse(result["amortized_transfer_cost_advantage_claim"])
        # Diagnostic and farther-source values have no effect on retrieval.
        rows[-1]["energy_hartree"] = float("nan")
        rows[-2]["energy_hartree"] = "not read"
        self.assertEqual(self.retrieve(rows), result)

    def test_no_farther_fallback_or_target_rows(self):
        self.assertEqual(self.retrieve([self.row("other", alias="CO2_T02")])["candidates"], [])
        for alias in ("CO2_I01", "CO2_V01"):
            with self.assertRaises(ValueError):
                self.retrieve([self.row("heldout", alias=alias, role="diagnostic")])
        extra = self.row("extra")
        extra["target_energy"] = -999
        with self.assertRaises(ValueError):
            self.retrieve([extra])
        with self.assertRaises(TypeError):
            self.retrieve([], target_energy=-999)

    def test_caps_duplicates_and_invalid_source_data(self):
        rows = [self.row(str(i)) for i in range(1024)]
        result = self.retrieve(rows)
        self.assertEqual(len(result["candidates"]), 1023)
        self.assertEqual(result["maximum_total_logical_requests"], 1024)
        self.assertTrue(result["duplicates_and_cache_hits_charged"])
        self.assertEqual(len(self.retrieve(rows, max_candidates=1)["candidates"]), 1)
        for bad in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                self.retrieve([self.row("bad", energy=bad)])
        with self.assertRaises(ValueError):
            self.retrieve([self.row("same"), self.row("same")])
        for cost in (None, {**self.cost, "logical_energy_requests": True},
                     {**self.cost, "wall_seconds": 0}, {**self.cost, "release_sha256": "fake"}):
            with self.assertRaises(ValueError):
                retrieval_candidates("CO2_I01", [], catalogue=self.catalogue, length=32, acquisition_cost=cost)


if __name__ == "__main__":
    unittest.main()
