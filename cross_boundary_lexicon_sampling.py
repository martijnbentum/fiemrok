'''Plan and sample cross-boundary types without changing phrase selections.'''

from collections import Counter
from dataclasses import dataclass
from heapq import heappop, heappush
from math import ceil
import random

from progressbar import ProgressBar, progressbar


class LengthBin:
    '''Count types within ±max(tolerance, 10% of the current target).'''

    def __init__(self, n_phones, target_count, tolerance=10):
        self.n_phones = n_phones
        self.target_count = target_count
        deviation = max(tolerance, target_count // 10)
        self.lower_bound = max(0, target_count - deviation)
        self.upper_bound = target_count + deviation
        self.remaining_types = []
        self.selected_types = []

    @property
    def n_selected_types(self):
        return len(self.selected_types)

    @property
    def below_minimum(self):
        return self.n_selected_types < self.lower_bound

    @property
    def full(self):
        return self.n_selected_types >= self.upper_bound


class SamplingPlan:
    '''Token count, tolerance, ideal distribution, and current quotas.'''

    def __init__(self, ideal_targets, eligible, tokens_per_type=100,
        tolerance=10, targets=None):
        validate_integer(tokens_per_type, 'tokens_per_type', minimum=1)
        self.tolerance = tolerance
        bins = make_length_bins(ideal_targets, tolerance)
        self.ideal_targets = {b.n_phones: b.target_count for b in bins}
        self.eligible = {n: eligible.get(n, 0) for n in self.ideal_targets}
        for count in self.eligible.values():
            validate_integer(count, 'eligible count')
        self.tokens_per_type = tokens_per_type
        if targets is None:
            targets = allocate_targets(self.ideal_targets, self.eligible,
                sum(self.ideal_targets.values()))
        if set(targets) != set(self.ideal_targets):
            raise ValueError('targets must use the ideal distribution lengths')
        for n, count in targets.items():
            validate_integer(count, 'target count')
            if count > self.eligible[n]:
                raise ValueError('target count exceeds eligible types')
        self.targets = dict(targets)

    @property
    def required_total(self):
        return sum(self.ideal_targets.values())

    @property
    def proposed_total(self):
        return sum(self.targets.values())

    @property
    def total_shortfall(self):
        return self.required_total - self.proposed_total

    def with_targets(self, targets):
        '''Return a revised plan, preserving the original ideal distribution.'''
        return SamplingPlan(self.ideal_targets, self.eligible,
            self.tokens_per_type, self.tolerance, targets)

    def __str__(self):
        lines = [f'Sampling plan: {self.tokens_per_type} tokens per type; '
            f'tolerance ±max({self.tolerance}, 10% of current quota).',
            ' Phones  Eligible  Ideal  Current  Minimum  Maximum']
        for b in make_length_bins(self.targets, self.tolerance):
            n = b.n_phones
            lines.append(f'{n:7d} {self.eligible[n]:9d} '
                f'{self.ideal_targets[n]:6d} {b.target_count:8d} '
                f'{b.lower_bound:8d} {b.upper_bound:8d}')
        lines.append(f'Ideal total: {self.required_total}; '
            f'current total: {self.proposed_total}; '
            f'reduction: {self.total_shortfall}')
        lines.append(describe_distortion(self.ideal_targets, self.targets))
        lines.append('Frequency only: overlap may still prevent sampling.')
        return '\n'.join(lines)


@dataclass
class SamplingResult:
    '''Successful provisional tokens plus reproducible sampling diagnostics.'''

    tokens: list
    seed: int
    attempts: int
    plan: SamplingPlan
    actual_counts: dict
    selected_attempt: int = 0

    @property
    def distortion_score(self):
        return distribution_loss(self.plan.ideal_targets, self.actual_counts)

    def __str__(self):
        lines = [f'Best sample: random_seed={self.seed}, selected on attempt '
            f'{self.selected_attempt}; searched {self.attempts} attempts.',
            f'Weighted distribution error: {self.distortion_score:.6f}.',
            ' Phones  Ideal  Quota  Actual  Change']
        for n, ideal in self.plan.ideal_targets.items():
            actual = self.actual_counts.get(n, 0)
            lines.append(f'{n:7d} {ideal:6d} {self.plan.targets[n]:6d} '
                f'{actual:7d} {actual - ideal:+7d}')
        lines.append(describe_distortion(self.plan.ideal_targets,
            self.actual_counts))
        return '\n'.join(lines)


class SamplingFailure(ValueError):
    '''Failed attempt or run; final errors expose plan and next_random_seed.'''

    def __init__(self, message, bins=None, plan=None, next_random_seed=None):
        super().__init__(message)
        self.counts = {} if bins is None else {
            b.n_phones: b.n_selected_types for b in bins}
        self.exhausted = set() if bins is None else {
            b.n_phones for b in bins if not b.remaining_types}
        self.plan = plan
        self.next_random_seed = next_random_seed



def inspect_lexicon_phonemes():
    '''Return length and phone counters from one pronunciation per word type.
    Load layer 6 from lexicon100_2466_cgn. Each word type contributes its
    most frequent token pronunciation; ties use lexical phone-label order.
    Print token/type counts, token counts per type, and one-phone types.
    Return (n_types_by_phone_count, phone_counts).
    '''
    from stores import _attach_cgn
    from validate_lexicon import load_metadata_layer, load_store_w2v

    pronunciations = {}
    embedding_store = load_store_w2v()
    try:
        _attach_cgn(embedding_store)
        metadatas = load_metadata_layer(embedding_store, 6)
        if not metadatas: raise ValueError('no lexicon metadata for layer 6')
        print('counting pronunciations in CGN lexicon layer 6')
        for metadata in progressbar(metadatas):
            word = metadata.phraser_object
            phones = sorted(word.phones, key=lambda phone: phone.start)
            pronunciation = tuple(phone.label for phone in phones)
            if not pronunciation:
                raise ValueError(f'no phones found for word {word.label}')
            counts = pronunciations.setdefault(word.label, Counter())
            counts[pronunciation] += 1
    finally:
        embedding_store.close()

    n_types_by_phone_count = Counter()
    phone_counts = Counter()
    tokens_per_type = Counter()
    for label, counts in pronunciations.items():
        tokens_per_type[sum(counts.values())] += 1
        pronunciation = min(counts, key=lambda p: (-counts[p], p))
        n_types_by_phone_count[len(pronunciation)] += 1
        phone_counts.update(pronunciation)
        if len(pronunciation) == 1:
            print(f'one-phone type: {label!r} -> {pronunciation[0]!r}')
    print(f'{len(metadatas)} tokens, {len(pronunciations)} word types')
    print(f'tokens per type -> number of types: {dict(tokens_per_type)}')
    return n_types_by_phone_count, phone_counts


def plan_sampling(lexicon, sequence_types_by_phone_count, tokens_per_type=100,
    tolerance=10):
    '''Plan from all frequent types, with no fixed capacity discount.
    lexicon:                       lexicon containing candidate tokens
    sequence_types_by_phone_count: ideal length distribution
    tokens_per_type:               exact number of occurrences per type
    tolerance:                     absolute floor for quota deviations
    Balance relative deviations from the ideal under frequency limits.
    Keep the ideal total if frequency capacity allows; otherwise report less.
    Existing selections do not affect the frequency diagnostic.
    '''
    validate_integer(tokens_per_type, 'tokens_per_type', minimum=1)
    bins = make_length_bins(sequence_types_by_phone_count, tolerance)
    candidates = lexicon.collect_candidate_tokens(tokens_per_type)
    eligible = count_eligible_types(candidates)
    ideal = {b.n_phones: b.target_count for b in bins}
    return SamplingPlan(ideal, eligible, tokens_per_type, tolerance)


def sample_lexicon(lexicon, plan, random_seed=0, tries_per_round=3,
    max_attempts=100, min_types=2200):
    '''Return a successful provisional result; never change phrase selection.
    Search the full attempt budget, retaining the lowest distribution error.
    Each round reduces the total by 5%, stopping at min_types, and adapts
    quotas using exhausted bins. Larger successes break equal-error ties.
    A final failure retains the current plan for explicit continuation.
    '''
    if not isinstance(plan, SamplingPlan):
        raise ValueError('plan must be a SamplingPlan')
    validate_integer(random_seed, 'random_seed', minimum=None)
    validate_integer(tries_per_round, 'tries_per_round', minimum=1)
    validate_integer(max_attempts, 'max_attempts', minimum=1)
    validate_integer(min_types, 'min_types')
    if plan.proposed_total < min_types:
        message = f'plan has {plan.proposed_total} types, below min_types='
        message += f'{min_types}; no sampling attempted'
        raise SamplingFailure(message, plan=plan, next_random_seed=random_seed)
    candidates = lexicon.collect_candidate_tokens(plan.tokens_per_type)
    eligible = count_eligible_types(candidates)
    if any(eligible.get(n, 0) < count for n, count in plan.targets.items()):
        raise ValueError('candidate pool changed; create a new sampling plan')
    attempts = 0
    current = plan
    best = None
    while attempts < max_attempts:
        print(f'\nQuota round: min_types={min_types}; '
            f'{max_attempts - attempts} attempts left')
        print(current)
        failures = []
        for _ in range(min(tries_per_round, max_attempts - attempts)):
            seed = random_seed + attempts
            attempts += 1
            print(f'attempt {attempts}/{max_attempts}, random_seed={seed}')
            try:
                tokens = sample_attempt(lexicon, current, candidates, seed)
            except SamplingFailure as error:
                failures.append(error)
                print(error)
                continue
            labels = {token.phoneme_sequence: token.n_phones
                for token in tokens}
            counts = dict(Counter(labels.values()))
            result = SamplingResult(tokens, seed, attempts, current, counts,
                selected_attempt=attempts)
            key = (result.distortion_score, -sum(counts.values()))
            if best is None or key < (best.distortion_score,
                -sum(best.actual_counts.values())):
                best = result
                print('New best successful sample:')
                print(result)
            else:
                print(f'Success with seed={seed}; retaining seed={best.seed}.')
        if attempts == max_attempts: break
        revised = relax_plan(current, failures, min_types)
        if revised.targets == current.targets:
            print('Quotas unchanged at min_types; continuing with new seeds.')
        else:
            print('Relaxing quotas against the original ideal distribution; '
                'observed exhaustion is not proof of capacity.')
            print(f'Total types: {current.proposed_total} -> '
                f'{revised.proposed_total}')
        current = revised
    if best is not None:
        best.attempts = attempts
        print('Search complete; retaining the best successful sample.')
        print(best)
        return best
    message = f'{failures[-1]}\nFailed {attempts} attempts; '
    message += 'selection unchanged.\n'
    message += f'Last constraints:\n{current}\n'
    message += f'To continue, use the error.plan with '
    message += f'random_seed={random_seed + attempts}.\n'
    message += 'Greedy failure does not prove that no valid sample exists.'
    raise SamplingFailure(message, plan=current,
        next_random_seed=random_seed + attempts)


def sample_attempt(lexicon, plan, tokens_by_type, seed):
    '''Try one seed using fresh bins and reservations, scarce bins first.'''
    rng = random.Random(seed)
    bins = make_length_bins(plan.targets, plan.tolerance)
    grouped = group_sequence_types_by_length(tokens_by_type, rng)
    for b in bins:
        b.remaining_types = grouped.get(b.n_phones, [])
    # Protect nonzero minimums first; equal scarcity favors longer sequences.
    bins.sort(key=lambda b: (b.lower_bound == 0,
        len(b.remaining_types) / max(1, b.target_count), -b.n_phones))
    reserved = {id(ps): bytearray(len(ps._occupied))
        for ps in lexicon.phrase_sequences}
    tokens = sample_types_from_bins(bins, tokens_by_type, plan.tokens_per_type,
        rng, reserved, minimums_only=True)
    tokens.extend(sample_types_from_bins(bins, tokens_by_type,
        plan.tokens_per_type, rng, reserved, minimums_only=False,
        ideal_targets=plan.ideal_targets))
    return tokens


def relax_plan(plan, failures, min_types):
    '''Rebalance from the original ideal while exploring a smaller total.
    Exhaustion only limits this round's capacity estimate. Later rounds can
    revisit bins; greedy failure is not a permanent capacity constraint.
    '''
    capacity = dict(plan.eligible)
    if failures:
        exhausted = set.intersection(*(f.exhausted for f in failures))
        for n in exhausted:
            observed = max(f.counts.get(n, 0) for f in failures)
            capacity[n] = min(capacity[n], observed)
    total = max(min_types, plan.proposed_total
        - ceil(plan.proposed_total / 20))
    # Preserve the requested total if the temporary estimates cannot supply it.
    # Treat exhaustion as a preference, rather than silently violating the floor.
    if sum(capacity.values()) < total:
        capacity = plan.eligible
    targets = allocate_targets(plan.ideal_targets, capacity, total)
    return plan.with_targets(targets)


def marginal_distribution_cost(n, count, ideal, total):
    '''Marginal weighted error, up to a common positive scale factor.
    The reference proportions always come from the original ideal. Scaling
    the ideal to the requested total makes smaller plans comparable.
    '''
    original_total = sum(ideal.values())
    expected = total * ideal.get(n, 0) / original_total if original_total else 0
    return (2 * count + 1 - 2 * expected) / max(ideal.get(n, 0), 1)


def distribution_loss(ideal, actual):
    '''Weighted squared error between normalized length distributions.
    Use one ideal type as the weight floor for zero-target lengths.
    '''
    original_total, total = sum(ideal.values()), sum(actual.values())
    if not original_total:
        return 0.0 if not total else float('inf')
    if not total: return float('inf')
    return sum((actual.get(n, 0) / total - ideal.get(n, 0) / original_total)**2
        / (max(ideal.get(n, 0), 1) / original_total)
        for n in set(ideal) | set(actual))


def allocate_targets(preferred, capacity, total):
    '''Minimize relative distribution error at an exact capacity-limited total.
    Allocate by increasing marginal cost of the separable convex objective.
    Equal costs favor longer bins. The original ideal must be preferred.
    '''
    targets = dict.fromkeys(preferred, 0)
    total = min(total, sum(capacity.get(n, 0) for n in targets))
    queue = []
    for n in targets:
        if capacity.get(n, 0):
            heappush(queue, (marginal_distribution_cost(n, 0, preferred,
                total), -n))
    for _ in range(total):
        _, negative_n = heappop(queue)
        n = -negative_n
        targets[n] += 1
        if targets[n] < capacity.get(n, 0):
            heappush(queue, (marginal_distribution_cost(n, targets[n],
                preferred, total), -n))
    return targets


def make_length_bins(sequence_types_by_phone_count, tolerance=10):
    '''Merge one-phone targets into two phones; retain zero-target lengths.'''
    validate_integer(tolerance, 'tolerance')
    targets = Counter()
    for n, count in sequence_types_by_phone_count.items():
        validate_integer(n, 'phone counts', minimum=1)
        validate_integer(count, 'type counts')
        targets[max(2, n)] += count
    return [LengthBin(n, count, tolerance)
        for n, count in sorted(targets.items())]


def count_eligible_types(candidates):
    return Counter(tokens[0].n_phones for tokens in candidates.values())


def validate_integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')


def describe_distortion(ideal, actual):
    '''Compare normalized distributions and report total loss separately.'''
    original_total, actual_total = sum(ideal.values()), sum(actual.values())
    if not original_total or not actual_total:
        return f'Empty distribution; distortion undefined; total {actual_total}.'
    lengths = set(ideal) | set(actual)
    variation = sum(abs(ideal.get(n, 0) / original_total
        - actual.get(n, 0) / actual_total) for n in lengths) / 2
    before = sum(n * count for n, count in ideal.items()) / original_total
    after = sum(n * count for n, count in actual.items()) / actual_total
    text = f'Types: {actual_total}/{original_total}; '
    text += f'distribution shifted: {variation:.2%}; '
    text += f'mean phones: {before:.3f} -> {after:.3f} ({after - before:+.3f}).'
    return text


def group_sequence_types_by_length(tokens_by_type, rng):
    '''Group type labels by number of phones and shuffle within each bin.'''
    types_by_length = {}
    for label in sorted(tokens_by_type):
        n_phones = tokens_by_type[label][0].n_phones
        types_by_length.setdefault(n_phones, []).append(label)
    for labels in types_by_length.values():
        rng.shuffle(labels)
    return types_by_length


def sample_types_from_bins(bins, tokens_by_type, tokens_per_type, rng,
    reserved_spans, minimums_only=False, ideal_targets=None):
    '''Fill minimums round-robin, then prioritize distribution improvement.
    First meet all lower bounds round-robin. Then choose each next bin by
    its marginal distribution cost against the original ideal.
    Modify bins and temporary occupancy only; return newly sampled tokens.
    Exhausted bins are skipped. Raise if a round cannot make progress.
    '''
    if minimums_only:
        target_total_types = sum(length_bin.lower_bound for length_bin in bins)
    else:
        target_total_types = sum(length_bin.target_count for length_bin in bins)
    n_selected_types = sum(length_bin.n_selected_types for length_bin in bins)
    sampled_tokens = []
    initial_selected_types = n_selected_types
    remaining_types = target_total_types - n_selected_types
    if not remaining_types: return sampled_tokens
    phase = 'bin minimums' if minimums_only else 'remaining total'
    print(f'sampling {remaining_types} types for {phase}')
    with ProgressBar(max_value=remaining_types) as bar:
        while n_selected_types < target_total_types:
            made_progress = False
            ordered = bins
            if not minimums_only:
                ideal = ideal_targets if ideal_targets is not None else {
                    b.n_phones: b.target_count for b in bins}
                ordered = sorted(bins, key=lambda b: (
                    marginal_distribution_cost(b.n_phones, b.n_selected_types,
                        ideal, target_total_types), -b.n_phones))
            for length_bin in ordered:
                if n_selected_types == target_total_types: break
                if length_bin.full: continue
                if minimums_only and not length_bin.below_minimum: continue
                chosen_tokens = sample_next_type_from_bin(length_bin,
                    tokens_by_type, tokens_per_type, rng, reserved_spans)
                if not chosen_tokens: continue
                sampled_tokens.extend(chosen_tokens)
                n_selected_types += 1
                made_progress = True
                bar.update(n_selected_types - initial_selected_types)
                if not minimums_only: break
            if not made_progress:
                message = format_sampling_failure(bins, tokens_per_type)
                raise SamplingFailure(message, bins=bins)
    return sampled_tokens


def format_sampling_failure(bins, tokens_per_type):
    '''Describe provisional counts and highlight unmet minimums.'''
    selected = sum(b.n_selected_types for b in bins)
    target = sum(b.target_count for b in bins)
    heading = f'Cannot sample {tokens_per_type} non-overlapping tokens/type.'
    lines = [heading,
        f'Types selected in this attempt: {selected} / {target} total.',
        'Counts below are types, not tokens. Shortfall is below the minimum.',
        '', ' Phones  Selected  Target  Minimum  Maximum  Shortfall']
    for b in sorted(bins, key=lambda b: b.n_phones):
        shortfall = max(0, b.lower_bound - b.n_selected_types)
        lines.append(f'{b.n_phones:7d} {b.n_selected_types:9d} '
            f'{b.target_count:7d} {b.lower_bound:8d} {b.upper_bound:8d} '
            f'{shortfall:10d}')
    lines.extend(['',
        'selection unchanged; provisional tokens were not committed.',
        'This greedy attempt failed; a valid selection may still exist.',
        'Try fewer tokens per type, lower type targets for failing lengths,',
        'more phrases, or another seed.'])
    return '\n'.join(lines)


def sample_next_type_from_bin(length_bin, tokens_by_type, n_tokens, rng,
    reserved_spans):
    '''Try unused types until one supplies n_tokens non-overlapping tokens.
    Record a successful type in the bin and reserve its temporary spans.
    Discard unsuccessful types; return [] if the bin runs out of types.
    '''
    while length_bin.remaining_types:
        label = length_bin.remaining_types.pop()
        chosen_tokens = sample_non_overlapping_tokens(tokens_by_type[label],
            n_tokens, rng, reserved_spans)
        if chosen_tokens:
            length_bin.selected_types.append(label)
            return chosen_tokens
    return []


def sample_non_overlapping_tokens(candidates, n_tokens, rng, reserved_spans):
    '''Return n_tokens occurrences and reserve their temporary phone spans.
    Try shuffled tokens greedily, without backtracking.
    Return [] without changing reserved_spans if this attempt falls short.
    Trial spans become shared reservations only when the full type succeeds.
    '''
    candidates = list(candidates)
    rng.shuffle(candidates)
    trial_spans = {}
    chosen_tokens = []
    for sequence in candidates:
        parent_id = id(sequence.parent)
        if parent_id not in trial_spans:
            trial_spans[parent_id] = bytearray(reserved_spans[parent_id])
        mask = trial_spans[parent_id]
        start, end = sequence.start_index, sequence.end_index
        if any(mask[start:end]): continue
        mask[start:end] = b'\x01' * sequence.n_phones
        chosen_tokens.append(sequence)
        if len(chosen_tokens) == n_tokens:
            reserved_spans.update(trial_spans)
            return chosen_tokens
    return []
