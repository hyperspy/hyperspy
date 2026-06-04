#!/bin/bash
# Wrapper shell that auto-activates the hyperspy-dev conda environment.
# Used by OpenCode's `shell` config so every bash invocation has the env active.
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate hyperspy-dev 2>/dev/null
exec /bin/bash "$@"
