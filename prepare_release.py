#!/usr/bin/env python3
import argparse
import re
import subprocess


def run_towncrier(tag):
    cmd = ("towncrier", "build", "--version", tag.strip("v"))

    return subprocess.call(cmd)


def update_fallback_version_in_pyproject(tag, fname="pyproject.toml"):
    version = tag.strip("v").split(".")
    if len(version) < 3:
        version += ["0"]

    # Default to +1 on patch version
    major, minor, patch = version[0], version[1], int(version[2]) + 1

    with open(fname, "r") as file:
        lines = file.readlines()

    pattern = "fallback_version"
    new_version = f"{major}.{minor}.{patch}.dev0"
    # Iterate through the lines and find the pattern
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            lines[i] = f'{pattern} = "{new_version}"\n'
            break

    # Write the updated content back to the file
    with open(fname, "w") as file:
        file.writelines(lines)

    print(
        f"\nNew (fallback) dev version ({new_version}) written to `pyproject.toml`.\n"
    )


if __name__ == "__main__":
    # Get tag argument
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    args = parser.parse_args()
    tag = args.tag

    # Update release notes
    run_towncrier(tag)

    # Update fallback version for setuptools_scm
    update_fallback_version_in_pyproject(tag)
