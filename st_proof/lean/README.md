# st_proof Lean project

A minimal Lean 4 + mathlib project to formalize view mergeability criteria.

- `View` models a simple arithmetic view (start, end, step).
- `IsCorrectlyMerged` formalizes exact-cover merging.
- Included: symmetry lemma; gcd-of-steps lemma stub for follow-up.

Usage (optional):

```bash
# ensure Lean 4 and Lake are installed
lake update
lake build
```
