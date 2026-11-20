'''Compare query tokens with other tokens on per-token features.

Each helper takes a feature table from token_features, query rows of shape
(n,) and other rows of shape (n,) or (n, k), such as a result's
neighbour_indices or same-type indices; it returns an array shaped like the
other rows. Float outputs use NaN where a comparison is undefined.
'''

from functools import lru_cache

from dutch_syllabifier import sonority
import numpy as np
from phraser.phonology.phone_features import get_feature_vector


def same_speaker(features, queries, others):
    '''Return True where both tokens come from the same speaker.'''
    return _equal(features, 'speaker', queries, others)


def same_recording(features, queries, others):
    '''Return True where both tokens come from the same recording.'''
    return _equal(features, 'recording', queries, others)


def same_component(features, queries, others):
    '''Return True where both tokens come from the same CGN component.'''
    return _equal(features, 'component', queries, others)


def same_gender(features, queries, others):
    '''Return 1.0 for equal and 0.0 for different gender, NaN if unknown.'''
    query, other = _values(features, 'gender', queries, others)
    known = np.isin(query, ('female', 'male')) & np.isin(other,
        ('female', 'male'))
    return np.where(known, (query == other).astype(float), np.nan)


def age_difference(features, queries, others):
    '''Return absolute age difference in years; NaN where age is 0.'''
    query, other = _values(features, 'age', queries, others)
    difference = np.abs(query - other).astype(float)
    return np.where((query == 0) | (other == 0), np.nan, difference)


def duration_ratio(features, queries, others):
    '''Return longer over shorter duration, so 1.0 means equal length.'''
    query, other = _values(features, 'duration_ms', queries, others)
    return np.maximum(query, other) / np.minimum(query, other)


def shared_left_word(features, queries, others):
    '''Return True where cross-boundary items share their left word.'''
    _check_cross_boundary(features)
    return _equal(features, 'left_word', queries, others)


def shared_right_word(features, queries, others):
    '''Return True where cross-boundary items share their right word.'''
    _check_cross_boundary(features)
    return _equal(features, 'right_word', queries, others)


def shared_previous_word(features, queries, others):
    '''Return 1.0 where the words just before both tokens match.

    NaN where either token has no preceding word: an empty context or the
    '*' padding at recording edges.
    '''
    return _shared_context(features, 'left_words', queries, others, -1)


def shared_next_word(features, queries, others):
    '''Return 1.0 where the words just after both tokens match; see above.'''
    return _shared_context(features, 'right_words', queries, others, 0)


def shared_context_word(features, queries, others):
    '''Return True where the tokens share any left or right context word.'''
    words = [_context_words(features, side, queries, others)
        for side in ('left_words', 'right_words')]
    (query_left, other_left), (query_right, other_right) = words
    pairs = zip(query_left.flat, query_right.flat, other_left.flat,
        other_right.flat)
    shared = [bool((set(a.split() + b.split()) & set(c.split() + d.split()))
        - {'*'}) for a, b, c, d in pairs]
    return np.asarray(shared).reshape(other_left.shape)


def absolute_difference(features, name, queries, others):
    '''Return the absolute difference of a numeric feature column.'''
    query, other = _values(features, name, queries, others)
    return np.abs(query.astype(float) - other.astype(float))


def phone_edit_distance(features, queries, others):
    '''Return Levenshtein distance between the tokens' phone sequences.'''
    return _sequence_distances(features, queries, others, _phone_distance)


def feature_edit_distance(features, queries, others):
    '''Return edit distance with substitution cost equal to the share of
    differing distinctive features; insertion and deletion cost 1.
    '''
    return _sequence_distances(features, queries, others, _feature_distance)


def sonority_distance(features, queries, others):
    '''Return edit distance over aligned sonority weights.

    Substitution costs the weight difference over the scale range, so equal
    sonority costs 0; insertion and deletion cost 1. NaN when a phone has
    no sonority class.
    '''
    return _sequence_distances(features, queries, others, _sonority_distance)


def random_pairs(n_tokens, n_pairs, seed=0, labels=None):
    '''Draw query and other rows uniformly, excluding self-pairs.

    labels: optional per-token labels; pairs sharing a label are redrawn,
            giving a baseline for wrong neighbours
    Returns (queries, others), both of shape (n_pairs,).
    '''
    if n_tokens < 2: raise ValueError('need at least two tokens')
    generator = np.random.default_rng(seed)
    queries = generator.integers(n_tokens, size=n_pairs)
    return queries, _draw_others(queries, n_tokens, generator, labels)


def random_others(queries, n_tokens, seed=0, labels=None):
    '''Draw one random other row per given query, excluding self.

    Keeps the queries fixed, so the baseline has the same query mix as the
    pairs it is compared with. labels: as in random_pairs.
    '''
    if n_tokens < 2: raise ValueError('need at least two tokens')
    generator = np.random.default_rng(seed)
    return _draw_others(np.asarray(queries), n_tokens, generator, labels)


def _draw_others(queries, n_tokens, generator, labels):
    '''Draw others uniformly, redrawing self and same-label pairs.'''
    others = np.empty(len(queries), dtype=np.int64)
    redraw = np.ones(len(queries), dtype=bool)
    while redraw.any():
        drawn = generator.integers(n_tokens - 1, size=redraw.sum())
        others[redraw] = drawn + (drawn >= queries[redraw])
        redraw = others == queries
        if labels is not None: redraw |= labels[queries] == labels[others]
    return others


def _values(features, name, queries, others):
    '''Return query values broadcast against the other rows' values.'''
    queries, others = np.asarray(queries), np.asarray(others)
    if others.shape[:1] != queries.shape or queries.ndim != 1:
        raise ValueError('queries must be (n,) and others (n,) or (n, k)')
    query = features[name][queries]
    if others.ndim == 2: query = query[:, None]
    return query, features[name][others]


def _equal(features, name, queries, others):
    query, other = _values(features, name, queries, others)
    return query == other


def _context_words(features, name, queries, others):
    '''Return query and other context strings with the other rows' shape.'''
    query, other = _values(features, name, queries, others)
    return np.broadcast_to(query, other.shape), other


def _shared_context(features, name, queries, others, position):
    '''Compare the context word at position; NaN where either is missing.'''
    query, other = _context_words(features, name, queries, others)
    query = np.array([_word_at(text, position) for text in query.flat],
        dtype=object)
    other = np.array([_word_at(text, position) for text in other.flat],
        dtype=object)
    missing = (query == '') | (other == '')
    shared = np.where(missing, np.nan, (query == other).astype(float))
    return shared.reshape(np.shape(others))


def _word_at(text, position):
    '''Return the context word at position; '' for padding or no context.'''
    word = text.split()[position] if text else ''
    return '' if word == '*' else word


def _check_cross_boundary(features):
    if not np.all(features['boundary_index'] > 0):
        raise ValueError('shared words require a cross-boundary table')


def _sequence_distances(features, queries, others, distance):
    '''Apply a cached string-pair distance to every query/other pair.'''
    query, other = _values(features, 'phones', queries, others)
    query = np.broadcast_to(query, other.shape)
    result = [distance(a, b) for a, b in zip(query.flat, other.flat)]
    return np.asarray(result, dtype=float).reshape(other.shape)


@lru_cache(maxsize=None)
def _phone_distance(a, b):
    return _edit_distance(a.split(), b.split(), _phone_cost)


@lru_cache(maxsize=None)
def _feature_distance(a, b):
    return _edit_distance(a.split(), b.split(), _feature_cost)


@lru_cache(maxsize=None)
def _sonority_distance(a, b):
    weights = [_sonority_weights(sequence) for sequence in (a, b)]
    if None in weights: return np.nan
    return _edit_distance(*weights, _sonority_cost)


def _phone_cost(a, b):
    return float(a != b)


@lru_cache(maxsize=None)
def _feature_cost(a, b):
    '''Share of differing distinctive features; 1 for unknown labels.'''
    if a == b: return 0.
    vectors = get_feature_vector(a), get_feature_vector(b)
    if None in vectors: return 1.
    return float(np.mean(np.not_equal(*vectors)))


def _sonority_cost(a, b):
    return abs(a - b) / (len(sonority.SONORITY_SCALE) - 1)


@lru_cache(maxsize=None)
def _sonority_weights(sequence):
    '''Return a tuple of sonority weights, or None for an unknown phone.'''
    try: return tuple(sonority.sonority_weights(sequence.split()))
    except ValueError: return None


def _edit_distance(a, b, substitution):
    '''Weighted Levenshtein distance with unit insertion and deletion.'''
    previous = [float(j) for j in range(len(b) + 1)]
    for i, x in enumerate(a, 1):
        current = [float(i)]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                previous[j - 1] + substitution(x, y)))
        previous = current
    return previous[-1]
