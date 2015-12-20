# -*- coding: utf-8 -*-

import os
from glob import glob
import shutil
from subprocess import call
import json
from urllib2 import urlopen
from urllib import urlretrieve

import winpython.wppm


def get_nsis_template_path():
    return os.path.join(os.path.abspath(os.path.split(__file__)[0]),
                        "NSIS_installer_script.nsi")


def get_nsis_plugins_path():
    return os.path.join(os.path.abspath(os.path.split(__file__)[0]),
                        "NSISPlugins")


def get_current_hyperspy_version():
    """Fetch version from pypi."""

    js = json.load(urlopen("https://pypi.python.org/pypi/hyperspy/json"))
    return js['info']['version']


def download_hyperspy_license():
    urlretrieve("https://raw.github.com/hyperspy/hyperspy/master",
                "COPYING.txt")


def create_delete_macro(path, name, add_uninstaller=True):
    """Create a NSIS macro to delete file structructure in path.
    """
    path = os.path.abspath(os.path.expanduser(path))
    skip = len(path) + 1
    lines = []
    lines.append("!macro %s INSTALL_PATH\n" % name)
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for filename in filenames:
            lines.append('\tDelete /REBOOTOK "%s"\n' % os.path.join("${INSTALL_PATH}",
                                                          dirpath[skip:],
                                                          filename))
            if os.path.splitext(filename)[1] == ".py":
                filename = os.path.splitext(filename)[0] + ".pyc"
                lines.append('\tDelete /REBOOTOK "%s"\n' % os.path.join("${INSTALL_PATH}",
                                                              dirpath[skip:],
                                                              filename))

        lines.append('\tRMDir /REBOOTOK "%s"\n' % os.path.join("${INSTALL_PATH}",
                                                     dirpath[skip:]))
        if add_uninstaller is True:
            lines.insert(-1, '\tDelete /REBOOTOK "%s"\n' % os.path.join(
                "${INSTALL_PATH}",
                dirpath[skip:],
                'Uninstall_Hyperspy_Bundle.exe'))
    lines.append("!macroend\n")
    with open(name + ".nsh", "w") as f:
        for line in lines:
            f.write(line)


class HSpyBundleInstaller:
    needed_packages = [
        'colorama',
        'configobj',
        'docutils',
        'ets',
        'formlayout',
        'guidata',
        'guiqwt',
        'h5py',
        'hyperspy',
        'ipython',
        'Jinja2',
        'logilab-astng',
        'logilab-common',
        'MarkupSafe',
        'matplotlib',
        'mock',
        'nose',
        'numba',
        'numexpr',
        'numpy',
        'Pillow',
        'pip',
        'Pygments',
        'pylint',
        'pyparsing',
        'PyQt',
        'PyQtdoc',
        'PyQwt',
        'pyreadline',
        'PySide',
        'python-dateutil',
        'pytz',
        'pywin32',
        'pyzmq',
        'scikit-image',
        'scikit-learn',
        'scipy',
        'seaborn',
        'setuptools',
        'simplejson',
        'six',
        'Sphinx',
        'spyder',
        'sympy',
        'traits',
        'traitsui',
        'tornado',
        'VTK',
        'winpython',
    ]

    def __init__(self, dist_path, arch=("32", "64")):
        """Tool to customize WinPython distributions to create the HyperSpy
        bundle installer for Windows.
        The "distribution path" must have the following structure:
        ├── packages2install
        │   ├── package1
        │   ├── package2
        │   └── ...
        └── WinPython-ARCH*
            ├── f1
            ├── f2
            └── ...
        Parameters
        ----------
        dist_path: string
            The path to the folder containing the WP distributions and all
            necessary files to create the HyperSpy Bundle distribution.
        """
        dist_path = os.path.abspath(os.path.expanduser(dist_path))
        self.dist_path = dist_path
        if not isinstance(arch, (list, tuple)):
            arch = (arch,)
        self.arch = arch
        self.wppath = dict((
            (a, glob(os.path.join(dist_path, "WinPython-%s*" % a))[0])
            for a in arch))
        self.distributions = dict((
            (a, winpython.wppm.Distribution(
                self.get_full_paths("python-*", a)))
            for a in arch))
        self.hspy_version = get_current_hyperspy_version()

    def get_full_paths(self, rel_path, arch):
        fp = glob(os.path.join(self.wppath[arch], rel_path))
        if len(fp) == 1:
            fp = fp[0]
        return fp

    def uninstall_unneeded_packages(self):
        print "Uninstalling unneeded packages."
        for distribution in self.distributions.values():
            for package in distribution.get_installed_packages():
                try:
                    if package.name not in self.needed_packages:
                        print "Uninstalling:", package.name
                except:
                    print("Uninstallation error")

    def remove_tools(self):
        for arch in self.arch:
            to_remove = []  # list(self.get_full_paths("Qt*", arch))
            if self.get_full_paths("TortoiseHg*", arch):
                to_remove.append(self.get_full_paths("TortoiseHg*", arch))
            for f in to_remove:
                print "Removing %s from WinPython %s bit" % (f, arch)
                os.remove(f)
            hg_dir = self.get_full_paths(
                os.path.join('tools', 'TortoiseHg'), arch)
            if hg_dir:
                shutil.rmtree(hg_dir)

    def install_local_packages(self):
        for arch in self.arch:
            if arch == "32":
                packages = glob(os.path.join(self.dist_path,
                                             "packages2install\\*win32*"))
            else:
                packages = glob(os.path.join(self.dist_path,
                                             "packages2install\\*amd64*"))
            packages += glob(os.path.join(self.dist_path,
                                          "packages2install\\*any*"))
            for package in packages:
                print("Installing %s" % package)
                try:
                    self.distributions[arch].install(
                        winpython.wppm.Package(package))
                except:
                    print("Error installing %s in WinPython %s bit " %
                          (package, arch))

    def install_pip_packages(self, packages):
        for wppath in self.wppath.values():
            for package in packages:
                print("Installing %s in %s" % (
                    package, wppath))
                call(['cmd.exe', '/C',
                      "%s\\WinPython Command Prompt.exe" % wppath,
                      "pip", "install", "--upgrade", package])

    def test_hyperspy(self):
        for wppath in self.wppath.values():
            call(['cmd.exe', '/C',
                  "%s\\WinPython Command Prompt.exe" % wppath,
                  "nosetests", "hyperspy"])

    def clean(self):
        """Remove all *.pyc and *.swp files"""
        for arch, wppath in self.wppath.iteritems():
            for dirpath, dirnames, filenames in os.walk(wppath):
                for fn in filenames:
                    if os.path.splitext(fn)[1] in (".swp", ".pyc"):
                        os.remove(os.path.join(dirpath, fn))

    def create_installers(self):
        """Create NSIS installer(s) from emplate."""
        for a in self.arch:
            with open(get_nsis_template_path(), 'r') as f, \
                    open('NSIS_installer_script-%sbit.nsi' % a, 'w') as fa:
                for i, line in enumerate(f):
                    if "__VERSION__" in line:
                        line = line.replace("__VERSION__",
                                            self.hspy_version)
                        fa.write(line)
                    elif "__ARCHITECTURE__" in line:
                        fa.write(line.replace("__ARCHITECTURE__", a + "bit"))
                    elif "__WINPYTHON_PATH__" in line:
                        fa.write(line.replace("__WINPYTHON_PATH__",
                                              self.get_full_paths("", a)))
                    elif "__PYTHON_FOLDER__" in line:
                        fa.write(line.replace(
                            "__PYTHON_FOLDER__",
                            os.path.split(
                                self.get_full_paths("python-*", a))[1]))
                    elif ";!define CL64 1" in line:
                        if a == '64':
                            fa.write(line[1:])
                    elif "__INSTALL_LOG__" in line:
                        fa.write(line.replace("__INSTALL_LOG__",
                                              self.get_log_name(int(a))))
                    elif "__NSIS_PLUGINS__" in line:
                        fa.write(line.replace("__NSIS_PLUGINS__",
                                              get_nsis_plugins_path()))
                    elif "__HSPY_ICON__" in line:
                        icons = self.get_full_paths(
                            "python-*\\Lib\\site-packages\\hyperspy\\data\\"
                            "hyperspy_bundle_installer.ico", a)
                        fa.write(line.replace("__HSPY_ICON__", icons))
                    elif "__DELETE_MACRO_NAME__" in line:
                        fa.write(line.replace("__DELETE_MACRO_NAME__",
                                              "hspy_delete" + a))
                    else:
                        fa.write(line)

    def create_delete_macros(self):
        for arch, wppath in self.wppath.iteritems():
            create_delete_macro(wppath,
                                "hspy_delete%s" % arch,
                                add_uninstaller=True)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        arch = sys.argv[2].split(',')
    else:
        arch = ("32", "64")
    if len(sys.argv) > 1:
        bundle_dir = sys.argv[1]
    else:
        bundle_dir = os.path.join(os.path.dirname(sys.executable),
                                  '..', '..')
    p = HSpyBundleInstaller(bundle_dir, arch)
    if len(sys.argv) > 3:
        p.hspy_version = sys.argv[3]
    p.uninstall_unneeded_packages()
    p.remove_tools()
    p.install_local_packages()
    p.create_delete_macros()
    p.create_installers()
