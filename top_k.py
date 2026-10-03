"""
top_k.py
--------
Top-K heavy-hitter tracker = Min-Heap of size K  +  Hash Map (query -> heap index).

The heap root is always the weakest of the current top K, so a new candidate
is compared against the root only:
    offer(query, count) -> O(log K)
    space               -> O(K)
The hash map lets us find a query already inside the heap in O(1) so we can
update its count in place instead of inserting a duplicate.
"""


class TopKTracker:
    def __init__(self, k):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.heap = []          # list of [count, query], min-heap ordered by count
        self.position = {}      # query -> its index inside self.heap

    def __len__(self):
        return len(self.heap)

    # ----- heap primitives (written by hand, no heapq) -----
    def _swap(self, i, j):
        heap = self.heap
        heap[i], heap[j] = heap[j], heap[i]
        self.position[heap[i][1]] = i
        self.position[heap[j][1]] = j

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i][0] >= self.heap[parent][0]:
                break
            self._swap(i, parent)
            i = parent

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            left, right, smallest = 2 * i + 1, 2 * i + 2, i
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    # ----- public API -----
    def offer(self, query, count):
        """Tell the tracker that `query` now has estimated frequency `count`."""
        if query in self.position:                 # already in top K: update in place
            i = self.position[query]
            self.heap[i][0] = count
            self._sift_down(i)                     # counts only grow -> may sink
        elif len(self.heap) < self.k:              # heap not full yet: insert
            self.heap.append([count, query])
            self.position[query] = len(self.heap) - 1
            self._sift_up(len(self.heap) - 1)
        elif count > self.heap[0][0]:              # beats the weakest: replace root
            del self.position[self.heap[0][1]]
            self.heap[0] = [count, query]
            self.position[query] = 0
            self._sift_down(0)

    def items(self):
        """Top-K as (query, count), most frequent first."""
        return [(q, c) for c, q in sorted(self.heap, key=lambda e: (-e[0], e[1]))]
