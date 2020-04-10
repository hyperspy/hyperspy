# Setup headless display
# https://github.com/travis-ci/travis-ci/issues/7313
sudo Xvfb :99 -ac -screen 0 1024x768x8 &
sleep 1 # give xvfb some time to start

# Setup conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda

