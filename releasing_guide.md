
Cut a Release
=============

Create a PR to the `RELEASE_next_patch` branch and go through the following steps:

**Preparation**
- Prepare the release by running the `prepare_release.py` python script (e.g. `python prepare_release.py 2.0.1`) , which will do the following:
  - update the release notes in `CHANGES.rst` by running `towncrier`,
  - update the `setuptools_scm` fallback version in `pyproject.toml`.
- Check release notes
- For a minor or major release, update the versioned documentation repository and documentation version switcher according to the instruction in the [developer guide](https://hyperspy.org/hyperspy-doc/current/dev_guide/writing_docs.html#hosting-versioned-documentation)
- (optional) check conda-forge and wheels build. Pushing a tag to a fork will run the release workflow without uploading to pypi
- Let that PR collect comments for a day to ensure that other maintainers are comfortable with releasing

**Tag and release**
:warning: this is a point of no return point :warning:
- push tag (`vx.y.z`) to the upstream repository and the following will be triggered:
  - build of the wheels and their upload to pypi
  - creation of a zenodo record and the mining of a DOI
  - creation of a Github Release
  - update of the current version on readthedocs to this release

**Post-release action**
- Merge the PR

Follow-up
=========

- Tidy up and close corresponding milestone
- A PR to the conda-forge feedstock will be created by the conda-forge bot

Additional information
======================

The version is defined by ``setuptools_scm`` at build time when packaging HyperSpy. In an editable installation (when using ``pip install -e .``), the version is defined from the
git repository.