
## 2026-06-04 PR1 Bugfixes Started
- Branch: pr1-bugfixes (based on origin/RELEASE_next_minor)
- 4 parallel agents dispatched:
  - bg_5cf457b6: lazy.py fixes (#3607, #3608, #3609, #3610, #3612)
  - bg_40202a71: _mva.py fixes (validation error, bss_model corruption)
  - bg_c0c8a6ff: _ornmf.py fix (#3611)
  - bg_3e418552: CHANGES.rst URL encoding fix

### Bug locations on baseline (origin/RELEASE_next_minor):
- lazy.py:1060 - coeff.map_blocks() result discarded
- lazy.py:1095 - `is False` instead of `= False`
- _ornmf.py:197,202 - negative mean → NaN
- _ornmf.py:282 - 1e9 vs 1e6 iteration bound, missing zero guard
- _mva.py:273 - AttributeError instead of ValueError
- _mva.py:1271+ - get_bss_model() mutates learning_results
