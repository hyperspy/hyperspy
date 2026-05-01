Fix two bugs in lazy signal decomposition:

- The result of ``coeff.map_blocks`` in Poissonian noise normalisation was
  discarded (no-op assignment); coefficients are now correctly rescaled.
- ``_unfolded4decomposition`` was not reset to ``False`` after lazy SVD,
  leaving the signal permanently in the unfolded state after decomposition.
