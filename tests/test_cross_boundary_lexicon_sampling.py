import contextlib
import io
import pathlib
import random
import sys
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cross_boundary_lexicon as cbl
import cross_boundary_lexicon_sampling as sampling
from test_cross_boundary_lexicon import make_phrase


def make_sampling_lexicon(type_counts, tokens_per_type=2):
    '''Build independent occurrences, keeping one full candidate per phrase.'''
    phrases = []
    lengths = []
    for n_phones, n_types in sorted(type_counts.items()):
        for type_index in range(n_types):
            labels = [f'p{n_phones}_{type_index}_{i}' for i in range(n_phones)]
            for token_index in range(tokens_per_type):
                phrases.append(make_phrase([['edge'] + labels[:-1],
                    [labels[-1], 'edge']], start=token_index * 1000))
                lengths.append(n_phones)
    lexicon = cbl.CrossBoundaryLexicon(phrases)
    for ps, n_phones in zip(lexicon.phrase_sequences, lengths):
        ps.sequences = [s for s in ps.sequences if s.n_phones == n_phones]
    return lexicon


class PlanningTests(unittest.TestCase):
    def test_tolerance_is_ten_or_ten_percent_and_configurable(self):
        for target, bounds in ((0, (0, 10)), (8, (0, 18)), (26, (16, 36)),
            (45, (35, 55)), (387, (349, 425))):
            b = sampling.LengthBin(4, target)
            self.assertEqual((b.lower_bound, b.upper_bound), bounds)
        b = sampling.LengthBin(4, 26, tolerance=2)
        self.assertEqual((b.lower_bound, b.upper_bound), (24, 28))

    def test_merges_one_phone_targets_and_retains_optional_lengths(self):
        bins = sampling.make_length_bins({1: 3, 2: 2, 12: 0})
        self.assertEqual({b.n_phones: b.target_count for b in bins},
            {2: 5, 12: 0})
        self.assertEqual(bins[1].lower_bound, 0)

    def test_no_capacity_discount_and_deficits_go_to_nearby_lengths(self):
        plan = sampling.SamplingPlan({2: 5, 3: 5, 4: 5}, {2: 20, 3: 20, 4: 2})
        self.assertEqual(plan.targets, {2: 5, 3: 8, 4: 2})
        self.assertEqual(plan.proposed_total, 15)
        self.assertEqual(plan.tokens_per_type, 100)
        self.assertIn('distribution shifted:', str(plan))

    def test_full_total_can_require_moving_beyond_original_upper_bounds(self):
        plan = sampling.SamplingPlan({2: 5, 12: 50}, {2: 100, 12: 0})
        self.assertEqual(plan.targets, {2: 55, 12: 0})
        self.assertEqual(plan.ideal_targets, {2: 5, 12: 50})

    def test_capacity_shortfall_is_reported(self):
        plan = sampling.SamplingPlan({2: 50, 3: 50}, {2: 20, 3: 30})
        self.assertEqual(plan.proposed_total, 50)
        self.assertEqual(plan.total_shortfall, 50)

    def test_empty_plan_and_invalid_inputs(self):
        self.assertEqual(sampling.SamplingPlan({}, {}).targets, {})
        for targets in ({0: 1}, {-1: 1}, {2.5: 1}, {2: -1}, {2: 1.5}):
            with self.assertRaises(ValueError):
                sampling.make_length_bins(targets)
        for tolerance in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                sampling.make_length_bins({2: 1}, tolerance)
        for tokens in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                sampling.SamplingPlan({2: 1}, {2: 1}, tokens_per_type=tokens)

    def test_public_planner_does_not_modify_selections(self):
        lexicon = make_sampling_lexicon({2: 8, 3: 3})
        plan = cbl.plan_sampling(lexicon, {2: 5, 3: 5}, tokens_per_type=2)
        self.assertIsInstance(plan, cbl.SamplingPlan)
        self.assertEqual(plan.targets, {2: 7, 3: 3})
        self.assertEqual(lexicon.selected_sequence_counts, {})
        self.assertIn('20.00%', sampling.describe_distortion(
            plan.ideal_targets, plan.targets))

    def test_allocation_keeps_total_and_frequency_limits(self):
        rng = random.Random(42)
        for _ in range(100):
            preferred = {n: rng.randrange(30) for n in range(2, 10)}
            capacity = {n: rng.randrange(30) for n in preferred}
            total = rng.randrange(200)
            result = sampling.allocate_targets(preferred, capacity, total,
                avoid={3, 5})
            self.assertEqual(sum(result.values()),
                min(total, sum(capacity.values())))
            self.assertTrue(all(0 <= result[n] <= capacity[n] for n in result))


class SamplingTests(unittest.TestCase):
    def assert_unselected(self, lexicon):
        self.assertEqual(lexicon.selected_sequence_counts, {})
        self.assertTrue(all(not any(ps._occupied)
            for ps in lexicon.phrase_sequences))

    def test_scarce_first_round_robin_has_exact_tokens_without_overlap(self):
        lexicon = make_sampling_lexicon({2: 8, 3: 6})
        plan = cbl.plan_sampling(lexicon, {2: 5, 3: 5}, tokens_per_type=2,
            tolerance=2)
        tokens = lexicon.sample_lexicon(plan, min_types=10)
        counts = Counter(t.phoneme_sequence for t in tokens)
        self.assertEqual(len(counts), 10)
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual([t.n_phones for t in tokens[::2]], [3, 2] * 5)
        occupied = set()
        for token in tokens:
            for index in range(token.start_index, token.end_index):
                self.assertNotIn((id(token.parent), index), occupied)
                occupied.add((id(token.parent), index))
        self.assertEqual(lexicon.selected_sequence_counts, counts)
        self.assertEqual(lexicon.sampling_result.actual_counts, {2: 5, 3: 5})

    def test_meets_minimums_before_optional_bins(self):
        lexicon = make_sampling_lexicon({2: 5, 3: 12})
        plan = cbl.plan_sampling(lexicon, {2: 2, 3: 10}, tokens_per_type=2,
            tolerance=2)
        tokens = lexicon.sample_lexicon(plan, min_types=12)
        self.assertEqual([t.n_phones for t in tokens[::2]][:8], [3] * 8)

    def test_failed_type_does_not_leak_reservations(self):
        lexicon = make_sampling_lexicon({2: 1}, tokens_per_type=1)
        token = lexicon.phrase_sequences[0].sequences[0]
        reserved = {id(token.parent): bytearray(len(token.parent._occupied))}
        original = {key: bytes(mask) for key, mask in reserved.items()}
        chosen = sampling.sample_non_overlapping_tokens([token, token], 2,
            random.Random(0), reserved)
        self.assertEqual(chosen, [])
        self.assertEqual({key: bytes(mask) for key, mask in reserved.items()},
            original)
        self.assert_unselected(lexicon)

    def test_overlap_failure_retries_without_committing_and_can_resume(self):
        phrases = [make_phrase([['edge', 'a', 'b'], ['c', 'edge']])
            for _ in range(2)]
        lexicon = cbl.CrossBoundaryLexicon(phrases)
        plan = cbl.plan_sampling(lexicon, {2: 1, 3: 1}, tokens_per_type=2)
        with self.assertRaises(sampling.SamplingFailure) as caught:
            lexicon.sample_lexicon(plan, random_seed=5, max_attempts=4,
                min_types=2)
        error = caught.exception
        self.assertEqual(error.next_random_seed, 9)
        self.assertIn('random_seed=9', str(error))
        self.assertEqual(error.plan.proposed_total, 2)
        self.assert_unselected(lexicon)
        # A lower total makes this pool feasible after adaptation.
        tokens = lexicon.sample_lexicon(error.plan,
            random_seed=error.next_random_seed, max_attempts=10, min_types=1)
        self.assertEqual(len(tokens), 2)
        self.assertGreater(lexicon.sampling_result.attempts, 3)

    def test_retries_keep_quotas_three_times_and_then_adapt(self):
        lexicon = make_sampling_lexicon({2: 8, 3: 8})
        plan = cbl.plan_sampling(lexicon, {2: 5, 3: 5}, tokens_per_type=2)
        original = sampling.sample_attempt
        calls = []

        def attempt(lexicon, current, candidates, seed):
            calls.append((seed, dict(current.targets)))
            tokens = original(lexicon, current, candidates, seed)
            if len(calls) <= 3:
                bins = sampling.make_length_bins(current.targets)
                for b in bins:
                    b.selected_types = list(range(2 if b.n_phones == 3 else 5))
                    b.remaining_types = [] if b.n_phones == 3 else ['spare']
                raise sampling.SamplingFailure('controlled exhaustion',
                    bins=bins)
            return tokens

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with mock.patch.object(sampling, 'sample_attempt',
                side_effect=attempt):
                tokens = lexicon.sample_lexicon(plan, random_seed=10,
                    max_attempts=5, min_types=9)
        self.assertEqual([seed for seed, _ in calls], [10, 11, 12, 13])
        self.assertTrue(all(targets == plan.targets
            for _, targets in calls[:3]))
        self.assertNotEqual(calls[3][1], plan.targets)
        self.assertEqual(lexicon.successful_seed, 13)
        self.assertIn('controlled exhaustion', output.getvalue())
        self.assertIn('Relaxing quotas', output.getvalue())
        signature = [(id(t.parent), t.start_index, t.end_index) for t in tokens]
        successful_plan = lexicon.sampling_result.plan
        lexicon.reset_selection()
        self.assertIsNone(lexicon.sampling_result)
        repeated = lexicon.sample_lexicon(successful_plan, random_seed=13,
            max_attempts=1, min_types=9)
        self.assertEqual(signature,
            [(id(t.parent), t.start_index, t.end_index) for t in repeated])
        self.assertEqual(plan.targets, {2: 5, 3: 5})

    def test_total_attempt_limit_and_seed_continuity_across_rounds(self):
        lexicon = make_sampling_lexicon({2: 8})
        plan = cbl.plan_sampling(lexicon, {2: 5}, tokens_per_type=2)
        with mock.patch.object(sampling, 'sample_attempt',
            side_effect=sampling.SamplingFailure('failure')) as attempt:
            with self.assertRaises(sampling.SamplingFailure) as caught:
                lexicon.sample_lexicon(plan, random_seed=20, max_attempts=8,
                    min_types=3)
        self.assertEqual([call.args[-1] for call in attempt.call_args_list],
            list(range(20, 28)))
        totals = [call.args[1].proposed_total
            for call in attempt.call_args_list]
        self.assertEqual(totals, [5, 5, 5, 4, 4, 4, 3, 3])
        self.assertEqual(caught.exception.next_random_seed, 28)
        self.assert_unselected(lexicon)

    def test_default_total_attempt_limit_is_one_hundred(self):
        lexicon = make_sampling_lexicon({2: 1})
        plan = cbl.plan_sampling(lexicon, {2: 1}, tokens_per_type=2)
        with contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.object(sampling, 'sample_attempt',
                side_effect=sampling.SamplingFailure('failure')) as attempt:
                with self.assertRaises(sampling.SamplingFailure) as caught:
                    lexicon.sample_lexicon(plan, min_types=1)
        self.assertEqual(attempt.call_count, 100)
        self.assertEqual(caught.exception.next_random_seed, 100)

    def test_invalid_controls_and_minimum_fail_before_attempting(self):
        lexicon = make_sampling_lexicon({2: 3})
        plan = cbl.plan_sampling(lexicon, {2: 2}, tokens_per_type=2)
        invalid = [dict(max_attempts=0), dict(tries_per_round=0),
            dict(random_seed=None), dict(min_types=-1)]
        for kwargs in invalid:
            with self.assertRaises(ValueError):
                lexicon.sample_lexicon(plan, **kwargs)
        with mock.patch.object(sampling, 'sample_attempt') as attempt:
            with self.assertRaisesRegex(ValueError, 'below min_types=2200'):
                lexicon.sample_lexicon(plan)
        attempt.assert_not_called()

    def test_unexpected_errors_are_not_retried(self):
        lexicon = make_sampling_lexicon({2: 3})
        plan = cbl.plan_sampling(lexicon, {2: 2}, tokens_per_type=2)
        with mock.patch.object(sampling, 'sample_attempt',
            side_effect=RuntimeError('unexpected')) as attempt:
            with self.assertRaisesRegex(RuntimeError, 'unexpected'):
                lexicon.sample_lexicon(plan, min_types=2)
        self.assertEqual(attempt.call_count, 1)

    def test_existing_selections_are_preserved(self):
        lexicon = make_sampling_lexicon({2: 3})
        plan = cbl.plan_sampling(lexicon, {2: 2}, tokens_per_type=2)
        token = lexicon.phrase_sequences[0].sequences[0]
        token.parent.select(token)
        with self.assertRaisesRegex(ValueError, 'reset selections'):
            lexicon.sample_lexicon(plan, min_types=2)
        self.assertEqual(token.parent.selected_sequences, [token])

    def test_empty_plan_can_sample_with_zero_minimum(self):
        lexicon = cbl.CrossBoundaryLexicon([])
        plan = cbl.plan_sampling(lexicon, {})
        self.assertEqual(lexicon.sample_lexicon(plan, min_types=0), [])

    def test_unreachable_minimum_does_not_commit_provisional_tokens(self):
        phrases = [make_phrase([['edge', 'a', 'b'], ['c', 'edge']])
            for _ in range(2)]
        lexicon = cbl.CrossBoundaryLexicon(phrases)
        plan = cbl.plan_sampling(lexicon, {2: 1, 3: 1}, tokens_per_type=2,
            tolerance=0)
        candidates = lexicon.collect_candidate_tokens(2)
        with self.assertRaises(sampling.SamplingFailure) as caught:
            sampling.sample_attempt(lexicon, plan, candidates, seed=0)
        self.assertEqual(sum(caught.exception.counts.values()), 1)
        self.assert_unselected(lexicon)

    def test_adaptation_recalculates_bounds_but_preserves_ideal(self):
        plan = sampling.SamplingPlan({2: 50, 12: 26}, {2: 100, 12: 26})
        bins = sampling.make_length_bins(plan.targets)
        for b in bins:
            b.selected_types = list(range(10 if b.n_phones == 12 else 50))
            b.remaining_types = [] if b.n_phones == 12 else ['spare']
        failure = sampling.SamplingFailure('exhausted', bins=bins)
        revised = sampling.relax_plan(plan, [failure] * 3, min_types=70)
        self.assertEqual(revised.targets, {2: 66, 12: 10})
        self.assertEqual(revised.ideal_targets, {2: 50, 12: 26})
        bounds = sampling.make_length_bins(revised.targets, revised.tolerance)
        self.assertEqual(bounds[1].lower_bound, 0)
        self.assertEqual(plan.targets, {2: 50, 12: 26})

    def test_quota_cycles_trigger_total_reduction(self):
        lexicon = make_sampling_lexicon({2: 6, 3: 6})
        plan = cbl.plan_sampling(lexicon, {2: 5, 3: 5}, tokens_per_type=2)
        other = plan.with_targets({2: 4, 3: 6})
        with mock.patch.object(sampling, 'sample_attempt',
            side_effect=sampling.SamplingFailure('failure')) as attempt:
            with mock.patch.object(sampling, 'relax_plan',
                side_effect=[other, plan]):
                with self.assertRaises(sampling.SamplingFailure):
                    lexicon.sample_lexicon(plan, tries_per_round=1,
                        max_attempts=3, min_types=8)
        totals = [call.args[1].proposed_total
            for call in attempt.call_args_list]
        self.assertEqual(totals, [10, 10, 9])

    def test_reference_inspection_remains_available_through_cbl(self):
        self.assertIs(cbl.inspect_lexicon_phonemes,
            sampling.inspect_lexicon_phonemes)

    def test_long_optional_bins_remain_eligible(self):
        lexicon = make_sampling_lexicon({2: 2, 12: 1})
        plan = cbl.plan_sampling(lexicon, {2: 2, 12: 1}, tokens_per_type=2)
        tokens = lexicon.sample_lexicon(plan, min_types=3)
        self.assertIn(12, [t.n_phones for t in tokens])

    def test_failure_table_explains_provisional_counts_and_shortfall(self):
        bins = [sampling.LengthBin(6, 26), sampling.LengthBin(7, 8)]
        bins[0].selected_types = list(range(12))
        message = sampling.format_sampling_failure(bins, 100)
        self.assertRegex(message, r'(?m)^\s*6\s+12\s+26\s+16\s+36\s+4$')
        self.assertIn('selection unchanged', message)


if __name__ == '__main__':
    unittest.main()
