"""IPython magic to switch the hyperspy plotting backend to anyplotlib."""


def _register_anyplotlib_magic(ip):
    """Register %anyplotlib magic with IPython."""
    from IPython.core.magic import register_line_magic

    @register_line_magic
    def anyplotlib(line):
        """Switch the hyperspy plotting backend to anyplotlib for this session.

        Usage
        -----
        %anyplotlib
        """
        from hyperspy.defaults_parser import preferences

        preferences.Plot.backend = "anyplotlib"
        print("hyperspy: switched plotting backend to anyplotlib")


def load_ipython_extension(ip):
    _register_anyplotlib_magic(ip)
