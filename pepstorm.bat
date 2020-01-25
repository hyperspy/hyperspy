@echo off
set root_dir=%~dp0
pushd %root_dir%
cd hyperspy/tests
for /D %%G in ("*") do (
  cd %%G
for /R %%G in ("*.py") do autopep8 --aggressive --in-place --max-line-length 130 %%G
cd ..
)
cd ../
for %%G in (_components _signals datasets docstrings io_plugins learn misc models samfire_utils utils) do (
cd %%G
for /R %%G in ("*.py") do autopep8 --aggressive --in-place --max-line-length 130 %%G
cd ..
)
popd
