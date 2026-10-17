# Review of the original `fast_fuzzy_search` ZIP

Reviewed the archive as delivered. Verdict up front: **the algorithm is correct,
the benchmark is not trustworthy, and one complexity claim is wrong.** The
implementation needed no bug fixes; the evidence around it did.

## 1. Correctness — passes, and that part was solid

The shipped test suite passes (4 tests). The randomized differential test against
a plain reference is the right idea and covers Unicode including an astral-plane
emoji.

I pushed it further with exhaustive enumeration rather than sampling — all
string pairs over small alphabets, every threshold:

| alphabet | maxlen | pairs | assertions | result |
|---|---|---|---|---|
| `{a,b}` | 5 | 3,969 | 19,845 | pass |
| `{a,b,c}` | 4 | 14,641 | 73,205 | pass |
| `{猫,狗,é,🙂}` | 3 | 7,225 | 28,900 | pass |
| `{a}` | 8 | 81 | 567 | pass |

Also verified: `k` larger than both strings, `k = -1`, empty on either side.
**No counterexample found.** The band-boundary logic — which is where this kind
of code usually breaks — is sound.

## 2. The benchmark measures something narrower than it claims

Three separate defects, in order of how much they distort the headline number.

**It reports a combined figure as a single optimization.** The baseline is
`levenshtein_reference(query, w) <= 2` with no length gate; the "optimized" arm
has the gate built in. So the printed `8.25x` mixes two changes. Splitting them
on the same data, same seed:

```
naive -> +length filter : 3.74x
+length filter -> banded : 2.20x
naive -> banded (total)  : 8.66x
```

Most of the reported speedup is the length gate, which is three lines of code
and is not what the README credits.

**`Matches: 0`.** Every candidate was rejected. The line
`assert set(reference_hits) == set(optimized_hits)` therefore compared two empty
sets — it printed as if it validated the run, but it could not have failed. Not a
bug in the code, but it is a correctness check that verifies nothing, sitting in
the one file a reader will trust for numbers.

**Correct but unrepresentative input.** Random 8-18 character strings against a
9 character query produce zero hits, so the benchmark never exercises the path
where matches exist — which is the entire purpose of a fuzzy matcher. It also
tests exactly one string length.

## 3. The real picture: the speedup runs from 0.48x to 140x

Same library, same host, varying only the candidate length and hit density:

| scenario | length | k | hits | naive vs banded |
|---|---|---|---|---|
| Chinese word lookup | 2-4 | 2 | 24 | **0.48x** |
| Chinese, short query | 1-4 | 2 | 1,686 | **0.82x** |
| English, dense hits | 5-8 | 2 | 67 | 0.85x |
| The shipped benchmark | 8-18 | 2 | 0 | 8.66x |
| Large threshold | 8-18 | 5 | 0 | 1.41x |
| Long strings | 20-40 | 2 | 0 | **92.5x** |

**On Chinese input the shipped code is roughly twice as slow as the naive loop
it replaces.** That is the case a Chinese-language user will hit first, and it is
absent from the report.

The mechanism: on 3-character strings the band is nearly the whole matrix, so it
saves almost no cells while tripling per-cell cost. On 30-character strings it
saves 90% of the cells and the early exit fires within a few rows.

## 4. Complexity claim is wrong

> "Runtime is O(k * min(n,m)) in the useful bounded case"

The band is `2k+1` wide over `n` rows, so the cost is `O(k * n)` where
`n = max(len(a), len(b))`, not `min`. It is correct only when the strings happen
to be the same length. Memory `O(k)` is right.

## 5. Small things

- `__pycache__/*.pyc` shipped in the archive; no `.gitignore`.
- `search_dict` relies on `Match.__dict__`, which works for a non-slotted frozen
  dataclass but breaks silently if someone adds `slots=True`.
- No `pyproject.toml`, so `pip install` and the test layout are implicit.
- `README` says "Python 3.10+ recommended" — accurate, since `int | None` is used
  in signatures.

## What changed in this directory

- **`fast_fuzzy.py`** — added a dispatcher: plain DP below `m = 4`, plain DP with
  row-minimum early exit below `6k+2`, banded above. Branch points measured, not
  guessed. Behaviour is bit-identical to the original (the exhaustive suite
  proves it). Fixes the 0.48x Chinese regression to ~0.95x of a hand-inlined
  scan, and keeps 136x on long strings.
- **`benchmark.py`** — rewritten as three arms (naive / +length filter / library)
  so `C/B` isolates banding. Five scenarios including Chinese and dense-hit data.
  Interleaved rounds, per-round ratio medians. Warns when a case produces zero
  hits instead of silently passing.
- **`tests/`** — 4 tests to 17: added exhaustive enumeration, the degenerate
  single-character case, dispatch-boundary checks on both sides, and API-level
  tests for pruning exactness and casefold length changes (`'ß'` vs `'ss'`).
- **`docs/`** — three figures plus `make_figures.py`, which regenerates them from
  live measurements. Everything is plotted as a ratio to the naive scan because
  absolute timings drifted 2× between runs on the same machine.
- **`README.md` / `OPTIMIZATION_REPORT.md`** — corrected `O(k*min)` to
  `O(k*max)`, replaced the single headline number with the measured range, and
  documented where the technique loses.
