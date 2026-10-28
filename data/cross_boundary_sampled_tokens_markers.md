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

`lexicon.make_markers(store, save=False)` converts the retained sampling
result using `phraser.make_markers`. It returns markers in sampled-token order
and raises `ValueError` if there is no successful sample. Pass the same
phraser store that contains the source phrases.

```python
markers = lexicon.make_markers(cgn, save=True,
    keys_path='svs_data/cross_boundary_marker_keys.pkl')

# Refresh snapshot query roots before querying cgn.markers.
cgn.refresh_query_roots()
```

With `save=True`, ordered marker keys are pickled after the database save
succeeds. The default key path is `svs_data/cross_boundary_marker_keys.pkl`;
choose a different path to keep another sample index, or pass `keys_path=None`
to skip it. Parent directories are created as needed. With `save=False`,
markers are returned for preview and no key file is written.

```python
import cross_boundary_lexicon as cbl

markers = cbl.load_markers(cgn)
# Or select another saved sample index:
# markers = cbl.load_markers(cgn, keys_path='svs_data/another_sample.pkl')
```

`load_markers` is a module function; it needs no lexicon or sampling result.
It bulk-loads the markers from the supplied store in key-file order.
`make_markers` remains an instance method because it uses the retained sample.
Markers share the database with other objects, distinguished by object-type
rank. No manifest is created.

## EchoFrame integration

Use each persisted marker's own key when computing and storing embeddings.
Never reuse the source phrase key: different tokens would otherwise share
the same model/layer/collar cache identity, and metadata would resolve to the
whole phrase. Register the containing phraser store with EchoFrame so that
`metadata.phraser_object` can load the marker later.

Verify marker embedding computation and save/reload linkage before adopting
this pipeline. The generic segment interface appears suitable, but this
integration has not yet been tested end to end.

## Phone reconstruction and repeat exports

The marker interval runs from the first phone's start to the last phone's
end. `marker.overlap_phone` reconstructs the phones through the source phrase.
An integration test checks exact phone keys before and after saving and
reopening a temporary database. Verify this on a small batch of real CGN
annotations before exporting the full sample, particularly where phones have
gaps, overlap, or unusual boundary timestamps.

Repeated calls create new identifiers and do not deduplicate existing
markers. Keep the returned markers or their keys for later reuse. Deleting
by phoneme label alone can affect other samples with the same type.
