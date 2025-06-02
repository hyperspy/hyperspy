from setuptools import setup


def custom_version_scheme(version):
    def guess_version(version, retain):
        parts = [int(i) for i in str(version.tag).split(".")[:retain]]
        # Add missing parts up to retain
        while len(parts) < retain:
            parts.append(0)
        parts[-1] += 1
        # Add missing parts
        while len(parts) < 3:
            parts.append(0)

        return ".".join(str(i) for i in parts)

    from setuptools_scm import get_version

    version_from_scm = get_version()
    if "dev" not in version_from_scm:
        # this is a tag version, return it
        # used when building dist and wheel
        return version_from_scm

    # On RELEASE_next_major, "retain" needs to be 1
    # On RELEASE_next_minor, "retain" needs to be 2
    # On RELEASE_next_patch, "retain" needs to be 3
    return version.format_next_version(guess_version, retain=2)


setup(use_scm_version={"version_scheme": custom_version_scheme})
