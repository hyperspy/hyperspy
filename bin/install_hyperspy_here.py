import _winreg
import sys
import os


def install_hyperspy_here():
    # First uninstall old HyperSpy context menu entries
    try:
        if sys.getwindowsversion()[0] < 6.:  # Older than Windows Vista:
            _winreg.DeleteKey(
                _winreg.HKEY_LOCAL_MACHINE,
                r'Software\Classes\Folder\Shell\HyperSpy_here\Command')
            _winreg.DeleteKey(
                _winreg.HKEY_LOCAL_MACHINE,
                r'Software\Classes\Folder\Shell\HyperSpy_here')
        else:  # Vista or newer
            _winreg.DeleteKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\shell\hyperspy_here\Command')
            _winreg.DeleteKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\shell\hyperspy_here')
            _winreg.DeleteKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\Background\shell\hyperspy_here\Command')
            _winreg.DeleteKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\Background\shell\hyperspy_here')
        uninstall_hyperspy_here()
    except:
        # The old entries were not present, so we do nothing
        pass

    # Install the context menu entries for the qtconsole and the IPython
    # notebook
    for env in ('qtconsole', 'notebook'):
        script = os.path.join(sys.prefix, 'Scripts', "hyperspy_%s.bat" % env)
        if sys.getwindowsversion()[0] < 6.:  # Before Windows Vista
            key = _winreg.CreateKey(
                _winreg.HKEY_LOCAL_MACHINE,
                r'Software\Classes\Folder\Shell\HyperSpy_%s_here' %
                env)
            _winreg.SetValueEx(
                key,
                "",
                0,
                _winreg.REG_SZ,
                "HyperSpy %s here" %
                env)
            key.Close()
            key = _winreg.CreateKey(
                _winreg.HKEY_LOCAL_MACHINE,
                r'Software\Classes\Folder\Shell\HyperSpy_%s_here\Command' %
                env)
            _winreg.SetValueEx(
                key,
                "",
                0,
                _winreg.REG_EXPAND_SZ,
                script +
                " \"%L\"")
            key.Close()
        else:  # Windows Vista and above
            key = _winreg.CreateKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\shell\hyperspy_%s_here' %
                env)
            _winreg.SetValueEx(
                key,
                "",
                0,
                _winreg.REG_SZ,
                "HyperSpy %s here" %
                env)
            key.Close()
            key = _winreg.CreateKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\shell\hyperspy_%s_here\Command' %
                env)
            _winreg.SetValueEx(
                key,
                "",
                0,
                _winreg.REG_EXPAND_SZ,
                script +
                " \"%L\"")
            key.Close()
            key = _winreg.CreateKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\Background\shell\hyperspy_%s_here' %
                env)
            _winreg.SetValueEx(
                key,
                "",
                0,
                _winreg.REG_SZ,
                "HyperSpy %s Here" %
                env)
            key.Close()
            key = _winreg.CreateKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\Background\shell\hyperspy_%s_here\Command' %
                env)
            _winreg.SetValueEx(key, "", 0, _winreg.REG_EXPAND_SZ, script)
            key.Close()

    print("HyperSpy here correctly installed")

if __name__ == "__main__":
    install_hyperspy_here()
