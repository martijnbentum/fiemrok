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
# Preview without writing.
markers = lexicon.make_markers(cgn)

# Save the previewed markers after checking them.
cgn.save_many(markers)

# Refresh snapshot query roots before querying cgn.markers.
cgn.refresh_query_roots()
```

Alternatively, `lexicon.make_markers(cgn, save=True)` creates and bulk-saves
in one call. Markers share the database with other objects, distinguished
by object-type rank. No manifest is created.

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
