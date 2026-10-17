"""Exhaustive + randomized equivalence tests.

Three layers:
  1. exhaustive small-alphabet enumeration (no sampling, no luck involved)
  2. seeded randomized differential test against a plain reference
  3. API-level behaviour: pruning, ranking, scores, Unicode casefolding
"""
import itertools
import random
import string
import unittest
from fast_fuzzy import levenshtein_bounded, levenshtein_reference, search, search_dict


def all_strings(alphabet, maxlen):
    for n in range(maxlen + 1):
        for tup in itertools.product(alphabet, repeat=n):
            yield ''.join(tup)


class DistanceTests(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(levenshtein_bounded('kitten', 'sitting', 3), 3)
        self.assertIsNone(levenshtein_bounded('kitten', 'sitting', 2))
        self.assertEqual(levenshtein_bounded('', 'abc', 3), 3)
        self.assertIsNone(levenshtein_bounded('', 'abc', 2))
        self.assertEqual(levenshtein_bounded('猫咪', '猫', 1), 1)
        self.assertEqual(levenshtein_bounded('', '', 0), 0)

    def test_exhaustive_binary(self):
        """All 63 strings over {a,b} up to length 5, all thresholds 0-4."""
        pool = list(all_strings('ab', 5))
        for a in pool:
            for b in pool:
                ref = levenshtein_reference(a, b)
                for k in range(5):
                    self.assertEqual(levenshtein_bounded(a, b, k),
                                     ref if ref <= k else None, (a, b, k, ref))

    def test_exhaustive_ternary(self):
        """All 121 strings over {a,b,c} up to length 4 — hits the band edges."""
        pool = list(all_strings('abc', 4))
        for a in pool:
            for b in pool:
                ref = levenshtein_reference(a, b)
                for k in range(4):
                    self.assertEqual(levenshtein_bounded(a, b, k),
                                     ref if ref <= k else None, (a, b, k, ref))

    def test_exhaustive_unicode(self):
        """BMP, precomposed Latin, and an astral-plane emoji."""
        pool = list(all_strings('猫狗é🙂', 3))
        for a in pool:
            for b in pool:
                ref = levenshtein_reference(a, b)
                for k in range(4):
                    self.assertEqual(levenshtein_bounded(a, b, k),
                                     ref if ref <= k else None, (a, b, k, ref))

    def test_single_char_repeats(self):
        """Degenerate case: band start advances every row while cells never vary."""
        pool = list(all_strings('a', 8))
        for a in pool:
            for b in pool:
                ref = levenshtein_reference(a, b)
                for k in range(7):
                    self.assertEqual(levenshtein_bounded(a, b, k),
                                     ref if ref <= k else None, (a, b, k, ref))

    def test_random_against_reference(self):
        rng = random.Random(20261017)
        alphabet = string.ascii_lowercase + '猫狗é🙂'
        for _ in range(3000):
            a = ''.join(rng.choice(alphabet) for _ in range(rng.randrange(0, 14)))
            b = ''.join(rng.choice(alphabet) for _ in range(rng.randrange(0, 14)))
            actual = levenshtein_reference(a, b)
            for k in range(0, 5):
                self.assertEqual(levenshtein_bounded(a, b, k),
                                 actual if actual <= k else None, (a, b, k, actual))

    def test_threshold_exceeds_lengths(self):
        for a, b in [('', ''), ('a', ''), ('abc', 'xyz'), ('', 'abcdefgh')]:
            self.assertEqual(levenshtein_bounded(a, b, 100), levenshtein_reference(a, b))

    def test_negative_threshold(self):
        self.assertIsNone(levenshtein_bounded('a', 'b', -1))
        self.assertIsNone(levenshtein_bounded('', '', -1))

    def test_dispatch_boundary(self):
        """Both DP paths must agree exactly at the switch-over length."""
        rng = random.Random(1234)
        for k in range(0, 5):
            for m in range(0, 2 * k + 4):
                for _ in range(40):
                    base = ''.join(rng.choice('abcde') for _ in range(m))
                    other = ''.join(rng.choice('abcde') for _ in range(rng.randrange(0, m + 3)))
                    self.assertEqual(levenshtein_bounded(base, other, k),
                                     (lambda d: d if d <= k else None)(levenshtein_reference(base, other)),
                                     (base, other, k))


class SearchTests(unittest.TestCase):
    def test_search(self):
        found = search('pyhton', ['python', 'java', 'pytorch', 'PYTHON'], max_distance=2, limit=3)
        self.assertEqual([x.value for x in found], ['PYTHON', 'python'])
        self.assertEqual(found[0].distance, 2)

    def test_validation(self):
        with self.assertRaises(ValueError):
            search('x', ['x'], max_distance=-1)
        with self.assertRaises(ValueError):
            search('x', ['x'], limit=-1)

    def test_limit_zero(self):
        self.assertEqual(search('a', ['a'], limit=0), [])

    def test_length_pruning_is_exact(self):
        """The length filter must not drop anything the full DP would accept."""
        rng = random.Random(77)
        words = [''.join(rng.choice('abc') for _ in range(rng.randrange(0, 8)))
                 for _ in range(400)]
        for q in ['', 'a', 'ab', 'abc', 'abcde']:
            for k in range(4):
                want = sorted(w for w in words if levenshtein_reference(q, w) <= k)
                got = sorted(m.value for m in search(q, words, max_distance=k, limit=len(words)))
                self.assertEqual(got, want, (q, k))

    def test_casefold_by_length_change(self):
        """'ß'.casefold() == 'ss' — lengths must be compared after folding."""
        found = search('ss', ['ß'], max_distance=0)
        self.assertEqual([m.value for m in found], ['ß'])
        self.assertEqual(found[0].distance, 0)

    def test_deterministic_tiebreak(self):
        r = search('ab', ['AB', 'ab', 'Ab', 'aB'], max_distance=0, limit=10)
        self.assertEqual([m.value for m in r], ['AB', 'Ab', 'aB', 'ab'])

    def test_empty_strings(self):
        found = search('', [''], max_distance=0)
        self.assertEqual(found[0].score, 100.0)

    def test_search_dict_shape(self):
        out = search_dict('pyhton', ['python'], max_distance=2, limit=1)
        self.assertEqual(set(out[0]), {'value', 'distance', 'score'})
        self.assertEqual(out[0]['value'], 'python')


if __name__ == '__main__':
    unittest.main()
