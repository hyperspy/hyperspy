# This only works with Sphinx >= 1.1

rm api/*.rst
sphinx-apidoc ../hyperspy -o api -s rst -f --private
