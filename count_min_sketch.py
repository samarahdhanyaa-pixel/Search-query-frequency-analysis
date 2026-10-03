"""
count_min_sketch.py
-------------------
Count-Min Sketch: a fixed-size d x w matrix of counters that estimates how
often each item appears in a stream.

    update(item)   -> O(d)   hash the item once per row, add 1 to one counter per row
    estimate(item) -> O(d)   hash again, return the MINIMUM of those d counters
    space          -> O(w*d) fixed, no matter how many distinct items arrive

Guarantee (Cormode & Muthukrishnan, 2005):
    true_count <= estimate <= true_count + eps * N      with probability >= 1 - delta
    where  w = ceil(e / eps)   and   d = ceil(ln(1 / delta))
"""

import math
import random
from array import array

_MERSENNE_PRIME = (1 << 61) - 1          # large prime for universal hashing
_FNV_OFFSET = 0xCBF29CE484222325         # FNV-1a 64-bit constants
_FNV_PRIME = 0x100000001B3
_MASK_64 = 0xFFFFFFFFFFFFFFFF


def fnv1a_64(text):
    """FNV-1a string hash. Written by hand because Python's built-in hash()
    is randomised on every run, which would make results non-repeatable."""
    h = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * _FNV_PRIME) & _MASK_64
    return h


class CountMinSketch:
    def __init__(self, width, depth, seed=42, conservative=False):
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        self.width = width
        self.depth = depth
        self.conservative = conservative
        self.total = 0                                   # N = items seen so far

        # The data structure itself: d rows, each an array of w 32-bit counters.
        self.table = [array("I", [0]) * width for _ in range(depth)]

        # One (a, b) pair per row gives d independent hash functions:
        #     h_i(x) = ((a_i * x + b_i) mod p) mod w
        rng = random.Random(seed)
        self._hash_params = [
            (rng.randrange(1, _MERSENNE_PRIME), rng.randrange(0, _MERSENNE_PRIME))
            for _ in range(depth)
        ]

    @classmethod
    def from_error_bounds(cls, epsilon, delta, **kwargs):
        """Pick w and d from the accuracy we want instead of guessing them."""
        width = math.ceil(math.e / epsilon)
        depth = math.ceil(math.log(1.0 / delta))
        return cls(width, depth, **kwargs)

    def _columns(self, item):
        """Column index of `item` in each of the d rows."""
        x = fnv1a_64(item)
        w = self.width
        return [((a * x + b) % _MERSENNE_PRIME) % w for a, b in self._hash_params]

    def update(self, item, count=1):
        """Record `count` occurrences of item. Returns the new estimate."""
        cols = self._columns(item)
        table = self.table
        self.total += count

        if not self.conservative:
            # Standard update: increment one counter in every row.
            smallest = None
            for row, col in zip(table, cols):
                row[col] += count
                if smallest is None or row[col] < smallest:
                    smallest = row[col]
            return smallest

        # Conservative update: only raise counters that are below the new
        # estimate. Collisions inflate counters less, so the error shrinks.
        target = min(row[col] for row, col in zip(table, cols)) + count
        for row, col in zip(table, cols):
            if row[col] < target:
                row[col] = target
        return target

    def estimate(self, item):
        """Estimated frequency of item (never below the true frequency)."""
        return min(row[col] for row, col in zip(self.table, self._columns(item)))

    def memory_bytes(self):
        """Bytes used by the counter matrix."""
        return sum(row.itemsize * len(row) for row in self.table)

    def error_bound(self):
        """Max overestimate promised by theory: (e / w) * N."""
        return math.e / self.width * self.total
