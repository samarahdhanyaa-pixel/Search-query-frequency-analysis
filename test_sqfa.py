"""
test_sqfa.py
------------
Unit tests.   Run:  python -m unittest -v test_sqfa
"""

import random
import unittest

import dataset
from count_min_sketch import CountMinSketch, fnv1a_64
from top_k import TopKTracker


def exact_counts(stream):
    counts = {}
    for item in stream:
        counts[item] = counts.get(item, 0) + 1
    return counts


class TestCountMinSketch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream = [dataset.normalize(q) for q in dataset.generate_stream(30_000, seed=7)]
        cls.exact = exact_counts(cls.stream)

    def build(self, **kwargs):
        sketch = CountMinSketch.from_error_bounds(0.001, 0.01, **kwargs)
        for q in self.stream:
            sketch.update(q)
        return sketch

    def test_empty_sketch_returns_zero(self):
        self.assertEqual(CountMinSketch(64, 4).estimate("anything"), 0)

    def test_single_item_is_exact(self):
        sketch = CountMinSketch(64, 4)
        for _ in range(25):
            sketch.update("weather today")
        self.assertEqual(sketch.estimate("weather today"), 25)
        self.assertEqual(sketch.total, 25)

    def test_dimensions_from_error_bounds(self):
        sketch = CountMinSketch.from_error_bounds(0.001, 0.01)
        self.assertEqual((sketch.width, sketch.depth), (2719, 5))

    def test_never_underestimates(self):
        sketch = self.build()
        for query, true in self.exact.items():
            self.assertGreaterEqual(sketch.estimate(query), true)

    def test_error_within_theoretical_bound(self):
        sketch = self.build()
        bound = sketch.error_bound()
        ok = sum(1 for q, c in self.exact.items() if sketch.estimate(q) - c <= bound)
        self.assertGreaterEqual(ok / len(self.exact), 0.99)

    def test_unseen_query_is_small(self):
        sketch = self.build()
        self.assertLessEqual(sketch.estimate("query nobody ever typed 12345"), sketch.error_bound())

    def test_conservative_update_never_underestimates_and_is_no_worse(self):
        standard, conservative = self.build(), self.build(conservative=True)
        for query, true in self.exact.items():
            self.assertGreaterEqual(conservative.estimate(query), true)
            self.assertLessEqual(conservative.estimate(query), standard.estimate(query))

    def test_memory_is_fixed(self):
        small, big = CountMinSketch(2719, 5), self.build()
        self.assertEqual(small.memory_bytes(), big.memory_bytes())
        self.assertEqual(big.memory_bytes(), 2719 * 5 * 4)

    def test_same_seed_gives_same_result(self):
        self.assertEqual(self.build().table, self.build().table)

    def test_bulk_count_update(self):
        sketch = CountMinSketch(128, 4)
        sketch.update("youtube", count=500)
        self.assertEqual(sketch.estimate("youtube"), 500)

    def test_rejects_invalid_size(self):
        with self.assertRaises(ValueError):
            CountMinSketch(0, 5)

    def test_hash_is_deterministic_and_spreads(self):
        self.assertEqual(fnv1a_64("chatgpt"), fnv1a_64("chatgpt"))
        self.assertNotEqual(fnv1a_64("chatgpt"), fnv1a_64("chatgpu"))


class TestTopK(unittest.TestCase):
    def test_keeps_the_k_largest(self):
        rng = random.Random(3)
        values = {f"q{i}": rng.randint(1, 10_000) for i in range(500)}
        tracker = TopKTracker(10)
        for query, count in values.items():
            tracker.offer(query, count)
        expected = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        self.assertEqual(tracker.items(), expected)

    def test_updates_existing_entry_without_duplicating(self):
        tracker = TopKTracker(3)
        for count in range(1, 50):
            tracker.offer("ipl live score", count)
        self.assertEqual(tracker.items(), [("ipl live score", 49)])

    def test_heap_property_holds(self):
        rng = random.Random(5)
        tracker = TopKTracker(15)
        running = {}
        for _ in range(5_000):
            q = f"q{rng.randint(0, 60)}"
            running[q] = running.get(q, 0) + 1
            tracker.offer(q, running[q])
        heap = tracker.heap
        for i in range(1, len(heap)):
            self.assertLessEqual(heap[(i - 1) // 2][0], heap[i][0])
        for query, index in tracker.position.items():
            self.assertEqual(heap[index][1], query)

    def test_fewer_items_than_k(self):
        tracker = TopKTracker(10)
        tracker.offer("a", 2)
        tracker.offer("b", 5)
        self.assertEqual(tracker.items(), [("b", 5), ("a", 2)])

    def test_sketch_plus_heap_matches_exact_top10(self):
        stream = [dataset.normalize(q) for q in dataset.generate_stream(30_000, seed=11)]
        sketch = CountMinSketch.from_error_bounds(0.001, 0.01)
        tracker = TopKTracker(10)
        for q in stream:
            tracker.offer(q, sketch.update(q))
        truth = sorted(exact_counts(stream).items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        self.assertEqual({q for q, _ in tracker.items()}, {q for q, _ in truth})


class TestDataset(unittest.TestCase):
    def test_normalize_merges_messy_variants(self):
        self.assertEqual(dataset.normalize("  IPL   Live score "), "ipl live score")

    def test_stream_is_reproducible_and_skewed(self):
        a, b = dataset.generate_stream(5_000), dataset.generate_stream(5_000)
        self.assertEqual(a, b)
        counts = exact_counts(dataset.normalize(q) for q in a)
        self.assertGreater(max(counts.values()), 50 * min(counts.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
