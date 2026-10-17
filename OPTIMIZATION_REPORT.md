# Optimization report

## Problem

A naive fuzzy-search loop computes the full Levenshtein distance for every
candidate, even when the caller only accepts matches within `k`. Full DP is
`O(n*m)` per pair.

Three standard optimizations apply, and they do not stack the way the textbook
suggests in Python:

1. **Length-difference gate.** If `abs(len(a) - len(b)) > k`, the distance
   cannot be `<= k`. Skip before touching the DP. Free, always correct.
2. **Banded DP.** Only cells with `|i - j| <= k` can lie on a path of cost
   `<= k`, so compute a diagonal band of width `2k+1`: `O(k*n)` time, `O(k)`
   memory.
3. **Row-minimum early exit.** If every cell in a row exceeds `k`, no later row
   can drop back down. Stop that candidate.

## The finding

In CPython, **optimization 2 is a net loss on short strings.** It reduces cell
count but raises per-cell cost: three index computations, two band-boundary
tests and three `min` arguments, against a tight loop that does neither.

Reported as speedup over the naive scan (`k=2`, 4,000 equal-length candidates,
median of per-round ratios):

| length | plain + early exit | banded | banded wins? |
|---|---|---|---|
| 2 | 0.92× | 0.49× | no — 2× slower than naive |
| 3 | 0.90× | 0.47× | no |
| 4 | 1.09× | 0.60× | no |
| 5 | 1.38× | 0.81× | no |
| 6 | 1.62× | 1.05× | no |
| 7 | 1.89× | 1.35× | no |
| 8 | 1.87× | 1.68× | no |
| 10 | 2.41× | 2.86× | marginally |
| 12 | 3.67× | 3.83× | marginally |
| 14 | 3.91× | 5.01× | yes |
| 17 | 4.88× | 8.23× | yes |
| 20 | 5.81× | 10.03× | yes |
| 26 | 7.12× | 16.53× | yes |
| 32 | 8.90× | 23.57× | yes |

The crossover sits around 9 characters for `k=2`, not at "large inputs". Below
it, plain DP with early exit is the best of the three everywhere.

That yields the three-way dispatch in `fast_fuzzy.py`:

- `m <= 3`: plain DP wins outright; the row count is too small for early exit to
  repay its per-row `min()`. (`k=0` routes to the band instead, where the band is
  one cell wide.)
- `3 < m < 6k + 2`: plain DP **with** early exit.
- `m >= 6k + 2`: banded.

**Why cut off at `6k+2 = 14` when the crossover measures at 9?** Because the
crossover moves by a character or two between runs and with hit density. The
cost of switching late is small (plain+early is within 4% of the band at 12
characters); the cost of switching early on a noisy measurement is a regression
on the short strings this library sees most. The cutoff is deliberately on the
safe side. `docs/make_figures.py` plots the measured band if you want to retune
it for your own data.

## Why the naive comparison overstates the result

The usual demonstration compares "full DP on every candidate" against "banded
DP" and reports one number. Two things distort it:

**The length gate is doing most of the work on short inputs.** With random
8-18 character candidates and a 9 character query, the gate alone rejects most
of the list before any DP runs. That is a `2-4x` win with three lines of code,
and it is not banding. Reporting the combined figure as "banding speedup" is a
misattribution.

**Zero-hit benchmarks verify nothing.** The original 5000-candidate run had
`Matches: 0` — both implementations returned empty lists, and the equality
assertion compared two empty sets. It passed while testing nothing. Correctness
belongs in a differential test against a reference implementation, which is what
`tests/` does: exhaustive over all 63 strings of length `<= 5` on a binary
alphabet, all 121 on a ternary alphabet, plus a Unicode set including an
astral-plane emoji, at every threshold.

## Correctness

`levenshtein_bounded` is checked against a plain reference on:

- all pairs of strings over `{a,b}` up to length 5, `k = 0..4` (19,845 assertions)
- all pairs over `{a,b,c}` up to length 4, `k = 0..3` (73,205 assertions)
- all pairs over `{猫,狗,é,🙂}` up to length 3 (28,900 assertions)
- single-character degenerate case, length up to 8
- 3,000 seeded random pairs, `k = 0..4`
- the exact dispatch boundary, `m` around `6k+2`, both sides
- `k` larger than both strings, `k = -1`, empty strings on either side

## Cost of the dispatcher

The adaptive version pays a small constant per call — a length comparison, an
equality check, one branch — that a hand-inlined scan for one specific shape
does not. Measured on 4000 Chinese words it costs about **5%** (`6.7 ms` vs
`6.4 ms`). That is the price of a general entry point, and it is dwarfed by the
`30x` it buys at the other end of the length range.

## Top-k tightening: implemented, and worth less than expected

`search()` used to scan everything, sort, then truncate. It now keeps a window of
the best `limit` results and tightens the working threshold to the current
`limit`-th distance as soon as the window fills. Tightening prunes in two places:
the length gate uses the new threshold, and the DP is called with it as `k`.
Results are byte-identical to the unpruned scan — this is exact, not approximate,
and `tests/test_fast_fuzzy.py` asserts that over ~1,900 randomized combinations.

Measured against the unpruned version (`benchmark.py`, part 2):

| corpus | limit | gain |
|---|---|---|
| Chinese 2-4 chars, no matches | 1 / 10 / 100 | 0.99× / 0.98× / 1.00× |
| Chinese 2-4 chars, near misses | 1 / 10 / 100 | 1.02× / 0.99× / 0.95× |
| English 20-40 chars, near misses | 1 / 10 / 100 | **1.34×** / 1.05× / 0.99× |

That is far less than the textbook argument suggests, and the reason is worth
stating: **the thing being skipped is already cheap.** By the time a candidate
reaches the DP, the length gate has usually removed it, and when it does reach
the DP the band plus early exit have already collapsed the work. There is little
left for a tighter threshold to remove.

The one place it pays is long strings with a small `limit`, where each skipped
candidate would have been genuinely expensive — and even there it is 1.3×, not
10×. With `limit` in the hundreds the window maintenance costs more than the
pruning saves, and the code falls back to appending unsorted for exactly that
reason.

It is kept because it is exact, cheap, and never loses more than a few percent,
not because it is a big win. Anyone reading a claim that top-k pruning gives a
large speedup on short-string fuzzy search should measure it themselves.

## What is not implemented

- **Indexing.** No BK-tree or n-gram prefilter. Every call is linear in
  `len(choices)`. This is the remaining big lever, and unlike the two above it
  changes the complexity class rather than the constant.
- **Normalization.** Callers should `unicodedata.normalize` their data; this
  module compares code points.

## Benchmark hygiene

`benchmark.py` interleaves the three arms within each round and reports medians
of per-round ratios, not ratios of medians. Background load and CPU frequency
drift hit all arms in a round together, so the ratio survives what the absolute
times do not. Absolute times are `best-of-N` and are useful only for scale.

`docs/make_figures.py` regenerates the three figures in `docs/`, and every figure
is a ratio for the same reason. During this project's own measurements, absolute
wall-clock times drifted by 2× between runs on the same machine while the ratios
held steady. A figure plotted in milliseconds would have been wrong; a figure
plotted in ratios was not.

