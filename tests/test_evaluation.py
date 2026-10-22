import math
import unittest

from exapt.evaluation import (
    disease_metrics, macro_metrics, macro_recall_at_k, rank_drugs,
)


class RankingTests(unittest.TestCase):
    def test_scores_descend_with_node_id_tie_breaking(self):
        scores = {3: 1.0, 2: 1.0, 1: 0.0}
        self.assertEqual(rank_drugs(scores, [1, 2, 3]), [2, 3, 1])

    def test_rejects_incomplete_or_nonfinite_candidate_scores(self):
        with self.assertRaisesRegex(ValueError, "candidate"):
            rank_drugs({1: 2.0}, [1, 2])
        for score in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(score=score):
                with self.assertRaisesRegex(ValueError, "finite"):
                    rank_drugs({1: score}, [1])


class RecallTests(unittest.TestCase):
    def test_macro_recall_weights_diseases_equally(self):
        rankings = {
            10: [2, 3, 1],
            11: [1, 2, 3],
        }
        positives = {
            10: {2, 1},
            11: {3},
        }
        # macro recall@2 = (1/2 + 0)/2 = 0.25
        self.assertEqual(macro_recall_at_k(rankings, positives, k=2), 0.25)


class RankingMetricTests(unittest.TestCase):
    def test_known_positions_have_exact_ap_ndcg_and_first_hit(self):
        scores = disease_metrics([2, 3, 1, 4], {1, 2})
        self.assertEqual(scores["recall@20"], 1.0)
        self.assertAlmostEqual(scores["map"], (1 + 2 / 3) / 2)
        self.assertAlmostEqual(
            scores["ndcg@20"],
            (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)),
        )
        self.assertEqual(scores["first_hit_rr"], 1.0)
        self.assertEqual(scores["hit@20"], 1.0)
        self.assertEqual(scores["observed_precision@20"], 0.1)

    def test_macro_metrics_weight_each_disease_equally(self):
        rankings = {10: [1, 2], 11: [1, 2]}
        positives = {10: {1, 2}, 11: {3}}
        values = macro_metrics(rankings, positives)
        self.assertEqual(values["recall@20"], 0.5)
        self.assertAlmostEqual(values["map"], 0.5)
        self.assertEqual(values["first_hit_rr"], 0.5)

    def test_unretrieved_indications_affect_ranking_denominators(self):
        # one known indication is retrieved, the other is absent
        scores = disease_metrics([1, 2, 3], {1, 4})
        self.assertEqual(scores["recall@5"], 0.5)
        self.assertEqual(scores["recall@20"], 0.5)
        self.assertEqual(scores["map"], 0.5)
        self.assertAlmostEqual(
            scores["ndcg@20"], 1 / (1 + 1 / math.log2(3))
        )
        self.assertEqual(scores["first_hit_rr"], 1.0)
        self.assertEqual(scores["observed_precision@20"], 0.05)

    def test_zero_hits_and_invalid_evaluation_inputs(self):
        scores = disease_metrics([1, 2, 3], {4})
        self.assertTrue(all(value == 0 for value in scores.values()))
        with self.assertRaisesRegex(ValueError, "observed indication"):
            disease_metrics([1, 2, 3], set())
        with self.assertRaisesRegex(ValueError, "Rankings"):
            macro_metrics({10: [1, 2]}, {11: {1}})
        with self.assertRaisesRegex(ValueError, "Invalid recall"):
            macro_recall_at_k({10: [1, 2]}, {10: {1}}, k=0)
