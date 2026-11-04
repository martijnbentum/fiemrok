# Embedding selectors

Add optional `model` and `condition` parameters to `build_embeddings()`.
Each defaults to all values; selecting both processes one Echoframe store.
Keep saved selection order and the existing layers, collar, batch controls,
return mapping, and missing-embedding resume behavior.

Create missing stores with the actual selection type count. Discover existing
stores by model/condition and open them through `lv_stores`. Reject mismatched
type counts, ambiguous directories, and invalid selectors. Load only requested
datasets. Keep the change scoped to the Python embedding entry point.

Verify individual and partial selectors, independence from an unselected key
file, invalid input, mismatch/ambiguity errors, and a real-store resume with
synthetic model outputs. Retain the default four-store/32-index integration
test. Document single-store and partial-selector calls.
