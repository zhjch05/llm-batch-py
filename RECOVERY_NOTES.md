# Recovery Notes

This tree was reconstructed from Codex session transcripts only.

Recovered exactly from session line dumps:
- Python sources under `src/llm_batch_py/` except `providers/__init__.py` which was reconstructed from transcript search output
- Rust source `src/lib.rs`
- README, docs, benchmark script, and most tests

Recovered with transcript-backed assumptions:
- `tests/__init__.py` as empty file
- `.python-version` as `3.12`

Not yet restored from transcripts in this pass:
- `Cargo.lock`
- `LICENSE`
- checked-in dSYM artifacts under `src/llm_batch_py/libllm_batch_py_core.dylib.dSYM/`

`tests/test_providers.py` is being restored separately from the recorded apply-patch history because simple line-range extraction was not sufficient for the latest version.
