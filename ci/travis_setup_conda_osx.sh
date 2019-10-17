# Setup headless display
# https://github.com/travis-ci/travis-ci/issues/7313
sudo Xvfb :99 -ac -screen 0 1024x768x8 &
sleep 1 # give xvfb some time to start

# Setup conda
curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p $HOME/miniconda

