"""Benchmarks for thresholded Levenshtein search.

Part 1 — three arms for the distance kernel
  A  naive      full DP, no length filter        (the pattern being improved)
  B  lenfilter  full DP + length-difference gate
  C  library    fast_fuzzy.levenshtein_bounded   (adaptive: plain or banded)

Reporting B is the point: C = B + banding, so B/C isolates what banding actually
buys. Comparing only A vs C (as the original benchmark did) credits the length
filter's win to the band, which overstates it by 2-4x.

Part 2 — end-to-end `search()` with and without top-k threshold tightening,
because that optimization does not pay uniformly either.

Method
  - arms interleaved within each round, per-round ratios, median reported
  - best-of-N wall clock per arm
  - hit counts asserted equal across arms; empty result sets are flagged, since
    a zero-hit comparison verifies nothing
"""
import random
import string
import statistics
import time

from fast_fuzzy import levenshtein_bounded, levenshtein_reference, search

ROUNDS = 5
N = 4000


def naive(q, words, k):
    return sum(1 for w in words if levenshtein_reference(q, w) <= k)


def lenfilter(q, words, k):
    lq = len(q)
    return sum(1 for w in words
               if abs(lq - len(w)) <= k and levenshtein_reference(q, w) <= k)


def library(q, words, k):
    return sum(1 for w in words if levenshtein_bounded(q, w, k) is not None)


def make_words(rng, n, lo, hi, alphabet):
    return [''.join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi))) for _ in range(n)]


ZH = ('的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产家电力里如水化高'
      '自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线'
      '内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题程展五果料象员革位'
      '入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回')
EN = string.ascii_lowercase


def case_zh():
    rng = random.Random(99)
    words = make_words(rng, N, 2, 4, ZH)
    return '计算机', words, 2


def case_short_en():
    rng = random.Random(11)
    words = make_words(rng, N, 5, 8, 'abcde')
    return 'abcade', words, 2


def typo(word, rng, edits=2):
    """Inject substitutions so the query has at least one guaranteed match."""
    chars = list(word)
    for _ in range(edits):
        i = rng.randrange(len(chars))
        chars[i] = rng.choice(EN)
    return ''.join(chars)


def case_mid_en():
    rng = random.Random(42)
    words = make_words(rng, N, 8, 18, EN)
    return typo(rng.choice(words), rng), words, 2


def case_long_en():
    rng = random.Random(3)
    words = make_words(rng, N, 20, 40, EN)
    return typo(rng.choice(words), rng), words, 2


def case_large_k():
    rng = random.Random(42)
    words = make_words(rng, N, 8, 18, EN)
    return typo(rng.choice(words), rng, edits=3), words, 5


CASES = [
    ('zh 2-4 chars, k=2', case_zh),
    ('en 5-8, k=2', case_short_en),
    ('en 8-18, k=2', case_mid_en),
    ('en 20-40, k=2', case_long_en),
    ('en 8-18, k=5', case_large_k),
]


def run(label, factory):
    q, words, k = factory()
    ta, tb, tc, hits = [], [], [], 0
    for _ in range(ROUNDS):
        t0 = time.perf_counter(); a = naive(q, words, k);  ta.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); b = lenfilter(q, words, k); tb.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); c = library(q, words, k); tc.append(time.perf_counter() - t0)
        assert a == b == c, (label, a, b, c)
        hits = a
    ba, bb, bc = min(ta), min(tb), min(tc)
    return dict(
        label=label, q=q[:24], k=k, hits=hits,
        a=ba, b=bb, c=bc,
        lenf=statistics.median([x / y for x, y in zip(ta, tb)]),
        band=statistics.median([y / z for y, z in zip(tb, tc)]),
        total=statistics.median([x / z for x, z in zip(ta, tc)]),
    )


def unpruned_search(query, choices, max_distance=2, limit=10):
    """`search()` without threshold tightening: scan, sort, truncate."""
    q = query.casefold()
    found = []
    for original in choices:
        cand = original.casefold()
        if abs(len(q) - len(cand)) > max_distance:
            continue
        d = levenshtein_bounded(q, cand, max_distance)
        if d is None:
            continue
        found.append((d, cand, original))
    found.sort(key=lambda t: (t[0], t[1], t[2]))
    return found[:limit]


def mutate(word, rng, n, alphabet):
    """Near-miss copies, so the corpus actually contains matches."""
    out = []
    for _ in range(n):
        chars = list(word)
        i = rng.randrange(len(chars))
        chars[i] = rng.choice(alphabet)
        out.append(''.join(chars))
    return out


def search_case_zh():
    rng = random.Random(99)
    words = make_words(rng, 6000, 2, 4, ZH)
    return 'short Chinese, query absent', '计算机', words, 2


def search_case_zh_rich():
    rng = random.Random(99)
    words = make_words(rng, 6000, 2, 4, ZH)
    q = '计算机'
    words += [q] + mutate(q, rng, 30, ZH[:80]) + mutate(q, rng, 30, ZH[:80])
    return 'short Chinese + near misses', q, words, 2


def search_case_en_long():
    rng = random.Random(5)
    base = ''.join(rng.choice(EN) for _ in range(30))
    words = make_words(rng, 3000, 20, 40, EN) + [base] + mutate(base, rng, 30, EN)
    return 'long English + near misses', base, words, 2


SEARCH_CASES = [
    ('zh 2-4, no matches', search_case_zh),
    ('zh 2-4, near misses', search_case_zh_rich),
    ('en 20-40, near misses', search_case_en_long),
]


def run_search(label, factory, limit):
    _tag, q, words, k = factory()
    tu, tp, hits = [], [], 0
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        a = unpruned_search(q, words, k, limit)
        tu.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        b = search(q, words, max_distance=k, limit=limit)
        tp.append(time.perf_counter() - t0)
        assert [(m.distance, m.value) for m in b] == [(d, v) for d, _, v in a], label
        hits = len(a)
    return dict(label=label, limit=limit, hits=hits,
                unpruned=min(tu), pruned=min(tp),
                gain=statistics.median([x / y for x, y in zip(tu, tp)]))


def main():
    print(f'candidates/arm: {N}   rounds: {ROUNDS}   arms interleaved per round')
    print('A=naive  B=+length filter  C=adaptive library')
    print()
    print(f"{'case':<20}{'hits':>6}{'A':>10}{'B':>10}{'C':>10}{'B/A':>8}{'C/B':>8}{'C/A':>8}")
    rows = []
    for label, factory in CASES:
        r = run(label, factory)
        rows.append(r)
        print(f"{r['label']:<20}{r['hits']:>6}{r['a']*1000:>9.1f}ms{r['b']*1000:>9.1f}ms"
              f"{r['c']*1000:>9.1f}ms{r['lenf']:>7.2f}x{r['band']:>7.2f}x{r['total']:>7.2f}x")
    print()
    empty = [r['label'] for r in rows if r['hits'] == 0]
    if empty:
        print(f"WARNING: zero hits in {empty} — those rows compare two empty result sets")
        print("         and therefore validate nothing about correctness.")
    print('Best-of-N wall clock; ratios are per-round medians. Absolute times are machine-specific.')
    print('C/B is the number that isolates banding. B/A is the length filter.')
    print()
    print('Reading the table: banding only pays above ~6*k+2 characters. Below that the library')
    print('dispatches to plain DP, so C/B sits near 1.0 instead of the 0.5x you get by forcing')
    print('the band on short strings. C/A is what a caller actually experiences.')

    print()
    print()
    print('=' * 88)
    print('search(): top-k threshold tightening vs an unpruned scan')
    print('=' * 88)
    print(f"{'case':<26}{'k':>3}{'limit':>7}{'hits':>6}{'unpruned':>12}{'pruned':>11}{'gain':>8}")
    for label, factory in SEARCH_CASES:
        for limit in (1, 10, 100):
            r = run_search(label, factory, limit)
            print(f"{r['label']:<26}{factory()[3]:>3}{r['limit']:>7}{r['hits']:>6}"
                  f"{r['unpruned']*1000:>10.1f}ms{r['pruned']*1000:>9.1f}ms{r['gain']:>7.2f}x")
    print()
    print('Tightening only pays where a skipped candidate would have been expensive:')
    print('long strings with a small limit. On 2-4 character Chinese it is a wash, and')
    print('with a limit in the hundreds the window maintenance eats the gain.')


if __name__ == '__main__':
    main()
