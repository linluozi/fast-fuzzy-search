"""Fast fuzzy search using bounded Levenshtein distance.

Pure Python, no third-party runtime dependencies. Unicode strings are supported.

The bounded (banded) DP wins big on long candidates and loses on short ones: at
small lengths the per-cell bookkeeping costs more than the band saves. This
module therefore dispatches on the shorter length: plain DP for very short
strings, plain DP with row-minimum early exit for medium ones, and the banded
DP only once the band covers a small enough slice of the matrix. See
OPTIMIZATION_REPORT.md for the measurements behind the branch points.
"""
from __future__ import annotations
import bisect
from dataclasses import dataclass
from typing import Iterable, Sequence

# Measured branch points on CPython 3.13 (see OPTIMIZATION_REPORT.md).
# Below _EARLY_EXIT_MIN_LEN the row count is too small for early exit to repay
# its per-row min() call. Below _BAND_SLOPE * k + 2 the band still covers most
# of the matrix, so plain DP with early exit wins; above it the band pays off.
_EARLY_EXIT_MIN_LEN = 4
_BAND_SLOPE = 6


def _full_distance(a: str, b: str) -> int:
    """Unbounded Levenshtein, two rolling rows, tight inner loop."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def levenshtein_reference(a: str, b: str) -> int:
    """Simple O(len(a)*len(b)) reference implementation for correctness tests."""
    return _full_distance(a, b)


def _levenshtein_plain_early(a: str, b: str, max_distance: int, m: int) -> int | None:
    """Full DP with row-minimum early exit; the workhorse for medium lengths.

    Caller guarantees len(a) >= len(b) == m >= 1. min() runs in C, which makes
    the per-row check far cheaper than tracking a running minimum in Python.
    """
    prev = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > max_distance:
            return None
        prev = cur
    result = prev[m]
    return result if result <= max_distance else None


def levenshtein_bounded(a: str, b: str, max_distance: int) -> int | None:
    """Return exact edit distance if <= max_distance, else None.

    Long inputs use a diagonal band of width 2*k+1 plus row-minimum early
    termination: O(k * max(n,m)) time, O(k) memory. Short inputs use a plain
    two-row DP, which is faster there (see module docstring).
    """
    if max_distance < 0:
        return None
    if a == b:
        return 0
    n, m = len(a), len(b)
    if abs(n - m) > max_distance:
        return None
    if n < m:
        a, b, n, m = b, a, m, n
    if m == 0:
        return n if n <= max_distance else None

    if m < _EARLY_EXIT_MIN_LEN and max_distance >= 1:
        d = _full_distance(a, b)
        return d if d <= max_distance else None
    if m < _BAND_SLOPE * max_distance + 2:
        return _levenshtein_plain_early(a, b, max_distance, m)

    return _levenshtein_banded(a, b, max_distance)


def _levenshtein_banded(a: str, b: str, max_distance: int) -> int | None:
    """Banded DP core: O(k * max(n,m)) time, O(k) memory.

    Exposed for benchmarking so `docs/make_figures.py` can plot the band on its
    own, unmediated by the dispatcher. Not part of the public API.
    """
    n, m = len(a), len(b)
    if n < m:
        a, b, n, m = b, a, m, n
    if m == 0:
        return n if n <= max_distance else None

    inf = max_distance + 1
    # Previous row is represented only within its active band.
    prev_start = 0
    prev = list(range(min(m, max_distance) + 1))
    for i in range(1, n + 1):
        start = max(0, i - max_distance)
        end = min(m, i + max_distance)
        cur = [inf] * (end - start + 1)
        row_min = inf
        for j in range(start, end + 1):
            idx = j - start
            if j == 0:
                value = i
            else:
                # deletion: D[i-1,j] ; insertion: D[i,j-1] ; substitution: D[i-1,j-1]
                deletion = prev[j - prev_start] + 1 if prev_start <= j <= prev_start + len(prev) - 1 else inf
                insertion = cur[idx - 1] + 1 if idx > 0 else inf
                substitution = (prev[j - 1 - prev_start] if prev_start <= j - 1 <= prev_start + len(prev) - 1 else inf) + (a[i - 1] != b[j - 1])
                value = min(deletion, insertion, substitution)
            cur[idx] = value
            if value < row_min:
                row_min = value
        if row_min > max_distance:
            return None
        prev, prev_start = cur, start
    result_index = m - prev_start
    result = prev[result_index] if 0 <= result_index < len(prev) else inf
    return result if result <= max_distance else None


@dataclass(frozen=True)
class Match:
    value: str
    distance: int
    score: float


def search(query: str, choices: Iterable[str], *, max_distance: int = 2,
           limit: int = 10, case_sensitive: bool = False) -> list[Match]:
    """Return closest strings, using cheap filters before bounded edit distance.

    Similarity score is 100 * (1 - distance / max_length), with empty/empty = 100.
    Ties are deterministic: distance, normalized value, then original value.

    The threshold starts at `max_distance` and tightens to the current k-th best
    distance once `limit` results are held. Tightening prunes twice: candidates
    farther than the new threshold skip the DP, and the length gate uses the new
    threshold too. Results are identical to an unpruned scan, not approximate.
    """
    if max_distance < 0:
        raise ValueError("max_distance must be >= 0")
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if limit == 0:
        return []

    q = query if case_sensitive else query.casefold()
    lq = len(q)
    # Sorted ascending by (distance, folded, original) — the same key the
    # unpruned implementation sort()s on, so ties break identically.
    kept: list[tuple[int, str, str]] = []
    threshold = max_distance

    for original in choices:
        if case_sensitive:
            candidate = original
            # Ties are broken on the folded form even when matching is
            # case-sensitive; that is what the unpruned version sorts on.
            tie = original.casefold()
        else:
            candidate = tie = original.casefold()
        if abs(lq - len(candidate)) > threshold:
            continue
        distance = levenshtein_bounded(q, candidate, threshold)
        if distance is None:
            continue
        entry = (distance, tie, original)
        if len(kept) < limit:
            # Append unsorted: with a large limit, inserting in order costs
            # O(limit) per hit for a list that will be sorted once at the end.
            kept.append(entry)
            if len(kept) == limit:
                kept.sort()
                threshold = kept[-1][0]
        elif entry < kept[-1]:
            bisect.insort(kept, entry)
            del kept[-1]
            threshold = kept[-1][0]

    if len(kept) < limit:
        kept.sort()

    result = []
    for distance, candidate, original in kept:
        denom = max(lq, len(candidate))
        score = 100.0 if denom == 0 else 100.0 * (1.0 - distance / denom)
        result.append(Match(original, distance, score))
    return result


def search_dict(query: str, choices: Sequence[str], *, max_distance: int = 2,
                limit: int = 10, case_sensitive: bool = False) -> list[dict[str, object]]:
    """Dictionary-shaped helper for JSON/API consumers."""
    return [m.__dict__.copy() for m in search(query, choices, max_distance=max_distance,
                                               limit=limit, case_sensitive=case_sensitive)]
