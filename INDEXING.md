# Indexing: one approach failed outright, the other has a narrow window

`search()` is a linear scan. This note records what happened when I tried to
replace it with an index, because the result was not what the literature
suggests and the two attempts failed differently.

## Attempt 1: BK-tree — failed

A BK-tree stores strings in a metric tree keyed by edit distance and prunes
subtrees using the triangle inequality: from a node at distance `d` from the
query, only children in the distance band `[d-k, d+k]` can hold matches.

It is the textbook answer for thresholded edit-distance search. Measurements on
80,000 strings:

| corpus | k | scan | BK query | speedup | nodes visited |
|---|---|---|---|---|---|
| Chinese 2-4 chars | 2 | 415 ms | 424 ms | 1.0× | 70,798 / 80,000 |
| Chinese 2-4 chars | 1 | 376 ms | 201 ms | 1.9× | 31,635 / 80,000 |
| English 8-18 | 2 | 314 ms | 782 ms | 0.4× | 28,568 / 80,000 |
| English 20-40 | 2 | 209 ms | 1,310 ms | 0.2× | 5,696 / 80,000 |

Two separate failures:

**Short strings do not prune.** With 2-4 character Chinese, almost any pair of
strings is within distance 2-4 of each other. The band `[d-k, d+k]` with `k=2`
therefore matches nearly every edge in the tree — 70,798 of 80,000 nodes are
visited. The triangle inequality is true but has no discriminating power when
the distance distribution is that compressed.

**Long strings prune well but pay too much per node.** At 20-40 characters the
tree visits only 5,696 nodes out of 80,000 — an excellent 14× reduction — and
still loses by 6×. The reason is that a BK-tree needs the *exact* distance to
navigate; `levenshtein_bounded` cannot be used, because `None` tells you nothing
about which subtree to descend. So every visited node costs a full unbounded DP,
while the linear scan's every candidate costs a length check plus a banded DP
that usually exits in three rows. **Navigation work is more expensive than the
work being skipped.**

Building is also slow: 28 s for 80,000 Chinese strings, 112 s for English 20-40,
because construction is `O(n)` exact distance computations per insert path.

## Attempt 2: q-gram prefilter — works, but only on long strings

A q-gram index maps every q-character substring to the strings containing it. If
`d(x, y) <= k`, then `y` must contain at least `|G(x)| - k*q` of `x`'s distinct
q-grams, because each edit can destroy at most `q` grams. That bound is cheap to
compute and, unlike a BK-tree, needs no exact distance to decide where to look.

At 20,000 strings:

| corpus | scan | build | indexed query | speedup | queries to break even |
|---|---|---|---|---|---|
| Chinese 2-4 chars | 96 ms | 30 ms | 101 ms | **0.95×** | never |
| English 8-18 | 114 ms | 132 ms | 6.0 ms | **18.8×** | 1 |
| English 20-40 | 72 ms | 238 ms | 18.6 ms | **3.9×** | 4 |

**Chinese short strings: the bound is zero.** A 3-character query yields two
2-grams. One edit destroys both. So `|G| - k*q = 2 - 4 = -2`, the bound is
vacuously true, `Index.search` detects that and falls back to a full scan — which
now costs 5% extra for the gram bookkeeping. This is not a tuning problem; there
is no `q` that fixes it. A 2-character query has one gram, a 4-character query
has three.

**English long strings: it is a real win.** At 14 characters the bound is
`13 - 4 = 9`, which most of the corpus fails, and only a handful of candidates
survive to the distance computation. 18.8× at 8-18 characters, 3.9× at 20-40
(the second is lower because each surviving candidate is expensive and there are
more of them).

## The line falls in the same place as the band's

Both findings land near the same threshold that governs the banded DP:

| string length | banded DP | q-gram index |
|---|---|---|
| 2-4 chars | loses (0.5×) | loses (0.95×) |
| 8-13 chars | marginal | 18.8× |
| 14+ chars | 4-24× | 3.9-18.8× |

The common cause is that **short strings have too little structure to exploit.**
A 3-character string has 9 matrix cells, 2 bigrams, and a distance distribution
that puts almost everything within 2 of everything else. There is no redundancy
to index and no gap to prune.

## What shipped

`Index` in `fast_fuzzy.py` — the q-gram version, since the BK-tree loses in
every case measured. It falls back to `search()` internally when the gram bound
cannot discriminate, so it is never *wrong*, only sometimes pointless. The
docstring says so, and `README` says so, because a class named `Index` invites
the assumption that it is faster.

Use it when all three hold:

1. the corpus is large (tens of thousands of strings or more),
2. the strings are longer than ~14 characters,
3. you query the same corpus repeatedly.

Otherwise `search()` is faster, and on Chinese word lookup it is the only
sensible choice.

## Reproducing

`benchmark.py` part 3 covers the shipped index. The BK-tree numbers above come
from a prototype in `_inspect_fast_fuzzy/probe_bktree.py` outside this repo; the
implementation is 40 lines of the same shape as `Index` and is not worth
shipping given the results.
