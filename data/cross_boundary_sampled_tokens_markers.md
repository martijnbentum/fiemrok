# Store sampled cross-boundary tokens as phraser markers

Use the existing `phraser.Marker` type to persist sampled tokens. A marker
represents a recording-relative time interval, has its own phraser key, and
can reference a phrase without belonging to its parent/child hierarchy.
There is no need to introduce a separate `Span` type.

## Conversion

- Persist only selected tokens, leaving candidates as lightweight
  `CrossBoundarySequence` objects.
- Set the marker label to the token's phoneme sequence.
- Copy start/end in milliseconds and the source audio.
- Explicitly supply the source phrase so its speaker and phrase link are
  used; do not rely on automatic overlap lookup.
- Bind and save the markers in the same phraser store as the source phrases.
- Do not reparent or copy the original words, syllables, or phones.

`phraser.make_markers(data, store, save=True)` supports bulk creation and
saving. Markers share the phraser database with other objects, distinguished
by their object-type rank. They are available through `store.markers`.

## EchoFrame integration

Use each persisted marker's own key when computing and storing embeddings.
Never reuse the source phrase key: different tokens would otherwise share
the same model/layer/collar cache identity, and metadata would resolve to the
whole phrase. Register the containing phraser store with EchoFrame so that
`metadata.phraser_object` can load the marker later.

Verify marker embedding computation and save/reload linkage before adopting
this pipeline. The generic segment interface appears suitable, but this
integration has not yet been tested end to end.

## Sample identity and provenance

Keep a sample manifest with the seed, quotas, tokens per type, filtering
settings, source phrase keys, and marker keys. Phones are assumed ordered
and non-overlapping. Reconstruct them using the marker interval from the
first phone's start to the last phone's end and `marker.overlap_phone`.
Separate exact phone references are not required under this assumption.

Deferred check: when implementing marker persistence, remind the user to
verify that `marker.overlap_phone` reproduces the sampled phones. This check
is intentionally deferred; no validation or persistence is implemented yet.

Define reuse of an existing marker for the same source phrase and phone span
before implementing repeat exports. Do not assume repeated construction is
idempotent: new segment objects receive new identifiers.

A sample collection identifier or manifest should distinguish sampling runs.
Phoneme labels identify types, not runs; deleting markers by label alone
could remove tokens belonging to other samples. Collection support is a
proposed addition, not an existing Marker field.
