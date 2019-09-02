# Configure conda
source $HOME/miniconda/bin/activate root
conda update -y conda
conda config --append channels conda-forge
conda create -n testenv --yes python=$PYTHON
conda activate testenv
# Install package with conda
conda install -y $DEPS $TEST_DEPS
conda info
