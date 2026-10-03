"""
main.py
-------
Search Query Frequency Analysis using a Count-Min Sketch.

    python main.py                 run the full demo on search_queries.txt
    python main.py --interactive   demo, then type your own queries to look up
    python main.py --file my.txt   use a different input file (one query per line)

Pipeline:  query log -> normalise -> Count-Min Sketch (+ Top-K min-heap)
                                  -> exact hash map (baseline, only to check accuracy)
"""

import argparse
import os
import sys
import time

import dataset
from count_min_sketch import CountMinSketch
from top_k import TopKTracker

EPSILON = 0.001      # allowed overestimate = 0.1 % of the stream length
DELTA = 0.01         # bound may be exceeded for at most 1 % of queries
TOP_K = 10


def hashmap_memory_bytes(counts):
    """Real memory of the exact baseline: the dict + every key string + every int."""
    return sys.getsizeof(counts) + sum(
        sys.getsizeof(key) + sys.getsizeof(value) for key, value in counts.items()
    )


def process_stream(queries, sketch, tracker=None):
    """Feed every query to the sketch (and tracker). Returns the exact counts too."""
    exact = {}
    for query in queries:
        estimate = sketch.update(query)
        if tracker is not None:
            tracker.offer(query, estimate)
        exact[query] = exact.get(query, 0) + 1
    return exact


def accuracy_report(sketch, exact):
    """Compare every estimate against the truth."""
    bound = sketch.error_bound()
    errors = [sketch.estimate(q) - true for q, true in exact.items()]
    return {
        "underestimates": sum(1 for e in errors if e < 0),
        "exact_hits": sum(1 for e in errors if e == 0),
        "avg_error": sum(errors) / len(errors),
        "max_error": max(errors),
        "bound": bound,
        "within_bound": sum(1 for e in errors if e <= bound) / len(errors),
    }


def line(char="-", width=74):
    print(char * width)


def run_demo(path, interactive=False):
    if not os.path.exists(path):
        if path != dataset.DEFAULT_PATH:
            sys.exit(f"Input file not found: {path}")
        print(f"{path} not found - generating it ...")
        dataset.write_file(path)

    sketch = CountMinSketch.from_error_bounds(EPSILON, DELTA)
    tracker = TopKTracker(TOP_K)

    start = time.perf_counter()
    exact = process_stream(dataset.read_file(path), sketch, tracker)
    elapsed = time.perf_counter() - start
    n = sketch.total
    if n == 0:
        sys.exit("Input file has no queries.")

    line("=")
    print("SEARCH QUERY FREQUENCY ANALYSIS  -  Count-Min Sketch")
    line("=")
    print(f"Input file          : {path}")
    print(f"Queries processed   : {n:,}")
    print(f"Distinct queries    : {len(exact):,}")
    print(f"Sketch size         : d = {sketch.depth} rows  x  w = {sketch.width:,} columns"
          f"   (eps = {EPSILON}, delta = {DELTA})")
    print(f"Processing time     : {elapsed:.2f} s   ({n / elapsed:,.0f} queries / second)")

    # ---------- 1. Top-K ----------
    print(f"\n[1] TOP {TOP_K} QUERIES   (min-heap + sketch  vs  exact hash map)")
    line()
    print(f"{'Rank':<5}{'Query':<30}{'Estimate':>10}{'Actual':>10}{'Error':>8}")
    line()
    true_top = sorted(exact.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K]
    for rank, (query, est) in enumerate(tracker.items(), start=1):
        print(f"{rank:<5}{query:<30}{est:>10,}{exact[query]:>10,}{est - exact[query]:>8}")
    overlap = len({q for q, _ in tracker.items()} & {q for q, _ in true_top})
    print(f"Top-{TOP_K} overlap with exact answer: {overlap} / {TOP_K}")

    # ---------- 2. Point queries ----------
    print("\n[2] FREQUENCY OF INDIVIDUAL QUERIES   (popular -> rare -> never searched)")
    line()
    print(f"{'Query':<38}{'Estimate':>10}{'Actual':>10}{'Error':>8}")
    line()
    ranked = sorted(exact.items(), key=lambda kv: (-kv[1], kv[0]))
    samples = [ranked[i][0] for i in (0, 24, 99, 499, 1999, len(ranked) - 1)]
    samples.append("zzz this was never searched")
    for query in samples:
        est, true = sketch.estimate(query), exact.get(query, 0)
        print(f"{query:<38}{est:>10,}{true:>10,}{est - true:>8}")

    # ---------- 3. Accuracy ----------
    report = accuracy_report(sketch, exact)
    print("\n[3] ACCURACY OVER ALL DISTINCT QUERIES")
    line()
    print(f"Underestimates              : {report['underestimates']}   (must be 0 - CMS only overestimates)")
    print(f"Exactly correct             : {report['exact_hits']:,} of {len(exact):,}")
    print(f"Average overestimate        : {report['avg_error']:.2f}")
    print(f"Maximum overestimate        : {report['max_error']}")
    print(f"Theoretical bound (eps * N) : {report['bound']:.0f}")
    print(f"Within bound                : {report['within_bound']:.2%}   (guaranteed >= {1 - DELTA:.0%})")

    # ---------- 4. Memory ----------
    cms_bytes, map_bytes = sketch.memory_bytes(), hashmap_memory_bytes(exact)
    print("\n[4] MEMORY")
    line()
    print(f"Exact hash map      : {map_bytes / 1024:>9,.1f} KB   (grows with every new distinct query)")
    print(f"Count-Min Sketch    : {cms_bytes / 1024:>9,.1f} KB   (fixed)")
    print(f"Top-K heap          : {TOP_K} entries")
    print(f"Saving              : {map_bytes / cms_bytes:.1f}x less memory")
    line("=")

    if interactive:
        print("\nType a search query to see its frequency (blank line to quit).")
        while True:
            try:
                raw = input("query> ")
            except EOFError:
                break
            query = dataset.normalize(raw)
            if not query:
                break
            est, true = sketch.estimate(query), exact.get(query, 0)
            print(f"   estimate = {est:,}    actual = {true:,}    error = {est - true}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search query frequency analysis with Count-Min Sketch")
    parser.add_argument("--file", default=dataset.DEFAULT_PATH, help="query log, one query per line")
    parser.add_argument("--interactive", action="store_true", help="look up your own queries after the demo")
    args = parser.parse_args()
    run_demo(args.file, args.interactive)
