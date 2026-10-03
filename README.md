# Search Query Frequency Analysis - Count-Min Sketch (Phase 3)

Needs only Python 3 (no extra libraries).

| Command | What it does |
|---|---|
| `python main.py` | Full demo on `search_queries.txt`: Top-10, point queries, accuracy, memory |
| `python main.py --interactive` | Same demo, then type any query to see estimate vs actual |
| `python main.py --file my.txt` | Run on your own file (one query per line) |
| `python -m unittest -v test_sqfa` | 19 unit tests |
| `python benchmark.py` | Efficiency experiments (memory, width, conservative update, time) |
| `python dataset.py` | Regenerate the input file |

## Files
- `count_min_sketch.py` - the sketch: 2-D counter array + d hash functions
- `top_k.py` - hand-written min-heap + hash map for Top-K queries
- `dataset.py` - builds / reads the 200,000-query input log
- `main.py` - demo and comparison against an exact hash map
- `benchmark.py`, `test_sqfa.py` - efficiency experiments and tests
- `search_queries.txt` - the input (200,000 queries, 14,135 distinct)
