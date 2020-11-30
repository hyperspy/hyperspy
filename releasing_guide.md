
Cut a Release
=============

Create a new PR to the `RELEASE_next_patch` and go through the following steps:
- Update and check changelog in `CHANGES.rst`
- Bump version in `hyperspy/Release.py`
- (optional) check conda-forge and wheels build
- Let that PR collect comments for a day to ensure that other maintainers are comfortable with releasing
- :warning: push tag (`vx.y.z`) to https://github.com/hyperspy/hyperspy, :warning: this is a point of no return point :warning:, because the following will be triggered:
  - creation of a zenodo record and the mining of a DOI
  - creation of a Github Release
  - build of the wheels and their upload to pypi
  - update of the current version on readthedocs to this release
- Increment the version and set it back to dev: `vx.y.z.dev0`
- Merge the PR

Follow-up
=========

- Prepare `CHANGES.rst` to add new entry
- Update version in other branches when necessary
- Tidy up and close corresponding milestone
- A PR to the conda-forge feedstock will be created by the conda-forge bot
