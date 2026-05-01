Fix several bugs in lazy signal decomposition:

- The result of ``coeff.map_blocks`` in Poissonian noise normalisation was
  discarded (no-op assignment); coefficients are now correctly rescaled.
- ``_unfolded4decomposition`` was not reset to ``False`` after lazy SVD,
  leaving the signal permanently in the unfolded state after decomposition.
- Navigation mask was not applied correctly for signals with
  multi-dimensional navigation spaces (2-D maps, etc.).
- ``_block_iterator`` only read the first signal chunk per navigation block
  when the on-disk chunk size was smaller than the full signal size (e.g.
  per-spectrum HDF5 chunking).  The signal dimension is now rechunked to a
  single chunk before iteration, so all signal channels are always read.
- ORNMF could hang indefinitely on data with a negative mean because
  ``np.sqrt(mean / m)`` produced ``NaN``, causing the convergence check to
  never trigger.  The absolute value of the mean is now used, and an
  off-by-three-orders-of-magnitude upper bound on the iteration count was
  corrected.
- NaN-fill guards for ``reproject`` results used a fragile string comparison
  instead of explicit boolean flags; reprojection failures (e.g. ORPCA/ORNMF
  signal reproject) could leave factors or loadings with the wrong shape.
