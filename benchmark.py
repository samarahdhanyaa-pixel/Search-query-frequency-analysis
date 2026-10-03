"""
benchmark.py
------------
Efficiency experiments on the same query log.

    python benchmark.py

  A. Memory vs stream length      - hash map keeps growing, sketch stays flat
  B. Accuracy vs sketch width     - doubling w roughly halves the error
  C. Standard vs conservative update
  D. Time per operation
"""

import time

import dataset
from count_min_sketch import CountMinSketch
from main import accuracy_report, hashmap_memory_bytes, process_stream

DEPTH = 5


def experiment_memory(queries):
    print("\nA. MEMORY vs STREAM LENGTH   (w = 2719, d = 5)")
    print(f"{'Queries':>10}{'Distinct':>10}{'HashMap KB':>12}{'CMS KB':>9}{'Saving':>9}")
    rows = []
    for n in (10_000, 25_000, 50_000, 100_000, 200_000):
        sketch = CountMinSketch(2719, DEPTH)
        exact = process_stream(queries[:n], sketch)
        map_kb = hashmap_memory_bytes(exact) / 1024
        cms_kb = sketch.memory_bytes() / 1024
        rows.append((n, len(exact), map_kb, cms_kb))
        print(f"{n:>10,}{len(exact):>10,}{map_kb:>12,.1f}{cms_kb:>9,.1f}{map_kb / cms_kb:>8.1f}x")
    return rows


def experiment_width(queries):
    print("\nB. ACCURACY vs WIDTH   (d = 5, all 200,000 queries)")
    print(f"{'Width':>7}{'CMS KB':>9}{'Avg err':>10}{'Max err':>9}{'Bound':>8}{'Top-100 avg err':>17}")
    rows = []
    for width in (256, 512, 1024, 2048, 4096, 8192):
        sketch = CountMinSketch(width, DEPTH)
        exact = process_stream(queries, sketch)
        report = accuracy_report(sketch, exact)
        top100 = sorted(exact.items(), key=lambda kv: -kv[1])[:100]
        top_err = sum(sketch.estimate(q) - c for q, c in top100) / 100
        rows.append((width, sketch.memory_bytes() / 1024, report["avg_error"], report["max_error"], report["bound"]))
        print(f"{width:>7,}{sketch.memory_bytes() / 1024:>9.1f}{report['avg_error']:>10.2f}"
              f"{report['max_error']:>9}{report['bound']:>8.0f}{top_err:>17.2f}")
    return rows


def experiment_conservative(queries):
    print("\nC. STANDARD vs CONSERVATIVE UPDATE   (w = 1024, d = 5)")
    print(f"{'Mode':<14}{'Avg err':>10}{'Max err':>9}{'Exactly correct':>18}")
    rows = []
    for name, flag in (("standard", False), ("conservative", True)):
        sketch = CountMinSketch(1024, DEPTH, conservative=flag)
        exact = process_stream(queries, sketch)
        report = accuracy_report(sketch, exact)
        rows.append((name, report["avg_error"], report["max_error"], report["exact_hits"]))
        print(f"{name:<14}{report['avg_error']:>10.2f}{report['max_error']:>9}"
              f"{report['exact_hits']:>11,} / {len(exact):,}")
    return rows


def experiment_time(queries):
    print("\nD. TIME PER OPERATION   (w = 2719)")
    print(f"{'Depth d':>8}{'Update (us)':>13}{'Query (us)':>12}")
    rows = []
    distinct = list(dict.fromkeys(queries))
    for depth in (1, 3, 5, 7, 9):
        sketch = CountMinSketch(2719, depth)
        start = time.perf_counter()
        for q in queries:
            sketch.update(q)
        update_us = (time.perf_counter() - start) / len(queries) * 1e6
        start = time.perf_counter()
        for q in distinct:
            sketch.estimate(q)
        query_us = (time.perf_counter() - start) / len(distinct) * 1e6
        rows.append((depth, update_us, query_us))
        print(f"{depth:>8}{update_us:>13.2f}{query_us:>12.2f}")
    print("Time grows with d only - it does not depend on how many queries were seen.")
    return rows


def run_all():
    queries = list(dataset.read_file())
    print(f"Loaded {len(queries):,} queries from {dataset.DEFAULT_PATH}")
    return {
        "memory": experiment_memory(queries),
        "width": experiment_width(queries),
        "conservative": experiment_conservative(queries),
        "time": experiment_time(queries),
    }


if __name__ == "__main__":
    run_all()
