# Configuration file for ipython-notebook.

from hyperspy import Release

# Configuration file for ipython.
c = get_config()
#c.TerminalIPythonApp.ignore_old_config = True
#c.TerminalInteractiveShell.banner2 = Release.info

#------------------------------------------------------------------------------
# NotebookApp configuration
#------------------------------------------------------------------------------

# NotebookApp will inherit config from: BaseIPythonApplication, Application

# Whether to prevent editing/execution of notebooks.
# c.NotebookApp.read_only = False

# Whether to enable MathJax for typesetting math/TeX
# 
# MathJax is the javascript library IPython uses to render math/LaTeX. It is
# very large, so you may want to disable it if you have a slow internet
# connection, or for offline use of the notebook.
# 
# When disabled, equations etc. will appear as their untransformed TeX source.
# c.NotebookApp.enable_mathjax = True

# Hashed password to use for web authentication.
# 
# To generate, type in a python/IPython shell:
# 
#   from IPython.lib import passwd; passwd()
# 
# The string should be of the form type:salt:hashed-password.
# c.NotebookApp.password = u''

# The full path to an SSL/TLS certificate file.
# c.NotebookApp.certfile = u''

# The url for MathJax.js.
# c.NotebookApp.mathjax_url = ''

# The IP address the notebook server will listen on.
# c.NotebookApp.ip = '127.0.0.1'

# Whether to create profile dir if it doesn't exist
# c.NotebookApp.auto_create = False

# The IPython profile to use.
c.NotebookApp.profile = u'hyperspy'

# Supply overrides for the tornado.web.Application that the IPython notebook
# uses.
# c.NotebookApp.webapp_settings = {}

# The full path to a private key file for usage with SSL/TLS.
# c.NotebookApp.keyfile = u''

# Whether to open in a browser after starting.
# c.NotebookApp.open_browser = True

# Set the log level by value or name.
# c.NotebookApp.log_level = 20

# Whether to install the default config files into the profile dir. If a new
# profile is being created, and IPython contains config files for that profile,
# then they will be staged into the new directory.  Otherwise, default config
# files will be automatically generated.
# c.NotebookApp.copy_config_files = False

# Create a massive crash report when IPython enconters what may be an internal
# error.  The default is to append a short message to the usual traceback
# c.NotebookApp.verbose_crash = False

# The name of the IPython directory. This directory is used for logging
# configuration (through profiles), history storage, etc. The default is usually
# $HOME/.ipython. This options can also be specified through the environment
# variable IPYTHON_DIR.
# c.NotebookApp.ipython_dir = u'/home/fjd29/.config/ipython'

# The port the notebook server will listen on.
# c.NotebookApp.port = 8888

# Whether to overwrite existing config files when copying
# c.NotebookApp.overwrite = False

#------------------------------------------------------------------------------
# IPKernelApp configuration
#------------------------------------------------------------------------------

# IPython: an enhanced interactive Python shell.

# IPKernelApp will inherit config from: KernelApp, BaseIPythonApplication,
# Application, InteractiveShellApp

# The importstring for the DisplayHook factory
# c.IPKernelApp.displayhook_class = 'IPython.zmq.displayhook.ZMQDisplayHook'

# Set the IP or interface on which the kernel will listen.
# c.IPKernelApp.ip = '127.0.0.1'

# Pre-load matplotlib and numpy for interactive use, selecting a particular
# matplotlib backend and loop integration.
c.IPKernelApp.pylab = "wx"

# Create a massive crash report when IPython enconters what may be an internal
# error.  The default is to append a short message to the usual traceback
# c.IPKernelApp.verbose_crash = False

# set the shell (XREP) port [default: random]
# c.IPKernelApp.shell_port = 0

# Whether to overwrite existing config files when copying
# c.IPKernelApp.overwrite = False

# Execute the given command string.
c.IPKernelApp.code_to_run = 'from hyperspy.hspy import *\n'

# set the stdin (XREQ) port [default: random]
# c.IPKernelApp.stdin_port = 0

# Set the log level by value or name.
# c.IPKernelApp.log_level = 30

# lines of code to run at IPython startup.
# c.IPKernelApp.exec_lines = []

# The importstring for the OutStream factory
# c.IPKernelApp.outstream_class = 'IPython.zmq.iostream.OutStream'

# Whether to create profile dir if it doesn't exist
# c.IPKernelApp.auto_create = False

# set the heartbeat port [default: random]
# c.IPKernelApp.hb_port = 0

# redirect stdout to the null device
# c.IPKernelApp.no_stdout = False

# dotted module name of an IPython extension to load.
# c.IPKernelApp.extra_extension = ''

# A file to be run
# c.IPKernelApp.file_to_run = ''

# The IPython profile to use.
c.IPKernelApp.profile = u'hyperspy'

# 
# c.IPKernelApp.parent_appname = u''

# kill this process if its parent dies.  On Windows, the argument specifies the
# HANDLE of the parent process, otherwise it is simply boolean.
# c.IPKernelApp.parent = 0

# JSON file in which to store connection info [default: kernel-<pid>.json]
# 
# This file will contain the IP, ports, and authentication key needed to connect
# clients to this kernel. By default, this file will be created in the security-
# dir of the current profile, but can be specified by absolute path.
# c.IPKernelApp.connection_file = ''

# If true, an 'import *' is done from numpy and pylab, when using pylab
# c.IPKernelApp.pylab_import_all = True

# The name of the IPython directory. This directory is used for logging
# configuration (through profiles), history storage, etc. The default is usually
# $HOME/.ipython. This options can also be specified through the environment
# variable IPYTHON_DIR.
# c.IPKernelApp.ipython_dir = u'/home/fjd29/.config/ipython'

# ONLY USED ON WINDOWS Interrupt this process when the parent is signalled.
# c.IPKernelApp.interrupt = 0

# Whether to install the default config files into the profile dir. If a new
# profile is being created, and IPython contains config files for that profile,
# then they will be staged into the new directory.  Otherwise, default config
# files will be automatically generated.
# c.IPKernelApp.copy_config_files = False

# List of files to run at IPython startup.
# c.IPKernelApp.exec_files = []

# A list of dotted module names of IPython extensions to load.
# c.IPKernelApp.extensions = []

# redirect stderr to the null device
# c.IPKernelApp.no_stderr = False

# set the iopub (PUB) port [default: random]
# c.IPKernelApp.iopub_port = 0

#------------------------------------------------------------------------------
# ZMQInteractiveShell configuration
#------------------------------------------------------------------------------

# A subclass of InteractiveShell for ZMQ.

# ZMQInteractiveShell will inherit config from: InteractiveShell

# Use colors for displaying information about objects. Because this information
# is passed through a pager (like 'less'), and some pagers get confused with
# color codes, this capability can be turned off.
# c.ZMQInteractiveShell.color_info = True

# 
# c.ZMQInteractiveShell.history_length = 10000

# Don't call post-execute functions that have failed in the past.
# c.ZMQInteractiveShell.disable_failing_post_execute = False

# Show rewritten input, e.g. for autocall.
# c.ZMQInteractiveShell.show_rewritten_input = True

# Set the color scheme (NoColor, Linux, or LightBG).
# c.ZMQInteractiveShell.colors = 'Linux'

# 
# c.ZMQInteractiveShell.separate_in = '\n'

# Deprecated, use PromptManager.in2_template
# c.ZMQInteractiveShell.prompt_in2 = '   .\\D.: '

# 
# c.ZMQInteractiveShell.separate_out = ''

# Deprecated, use PromptManager.in_template
# c.ZMQInteractiveShell.prompt_in1 = 'In [\\#]: '

# Enable deep (recursive) reloading by default. IPython can use the deep_reload
# module which reloads changes in modules recursively (it replaces the reload()
# function, so you don't need to change anything to use it). deep_reload()
# forces a full reload of modules whose code may have changed, which the default
# reload() function does not.  When deep_reload is off, IPython will use the
# normal reload(), but deep_reload will still be available as dreload().
# c.ZMQInteractiveShell.deep_reload = False

# Make IPython automatically call any callable object even if you didn't type
# explicit parentheses. For example, 'str 43' becomes 'str(43)' automatically.
# The value can be '0' to disable the feature, '1' for 'smart' autocall, where
# it is not applied if there are no more arguments on the line, and '2' for
# 'full' autocall, where all callable objects are automatically called (even if
# no arguments are present).
# c.ZMQInteractiveShell.autocall = 0

# 
# c.ZMQInteractiveShell.separate_out2 = ''

# Deprecated, use PromptManager.justify
# c.ZMQInteractiveShell.prompts_pad_left = True

# 
# c.ZMQInteractiveShell.readline_parse_and_bind = ['tab: complete', '"\\C-l": clear-screen', 'set show-all-if-ambiguous on', '"\\C-o": tab-insert', '"\\C-r": reverse-search-history', '"\\C-s": forward-search-history', '"\\C-p": history-search-backward', '"\\C-n": history-search-forward', '"\\e[A": history-search-backward', '"\\e[B": history-search-forward', '"\\C-k": kill-line', '"\\C-u": unix-line-discard']

# Enable magic commands to be called without the leading %.
# c.ZMQInteractiveShell.automagic = True

# 
# c.ZMQInteractiveShell.debug = False

# 
# c.ZMQInteractiveShell.object_info_string_level = 0

# 
# c.ZMQInteractiveShell.ipython_dir = ''

# 
# c.ZMQInteractiveShell.readline_remove_delims = '-/~'

# Start logging to the default log file.
# c.ZMQInteractiveShell.logstart = False

# The name of the logfile to use.
# c.ZMQInteractiveShell.logfile = ''

# 
# c.ZMQInteractiveShell.wildcards_case_sensitive = True

# Save multi-line entries as one entry in readline history
# c.ZMQInteractiveShell.multiline_history = True

# Start logging to the given file in append mode.
# c.ZMQInteractiveShell.logappend = ''

# 
# c.ZMQInteractiveShell.xmode = 'Context'

# 
# c.ZMQInteractiveShell.quiet = False

# Deprecated, use PromptManager.out_template
# c.ZMQInteractiveShell.prompt_out = 'Out[\\#]: '

# Set the size of the output cache.  The default is 1000, you can change it
# permanently in your config file.  Setting it to 0 completely disables the
# caching system, and the minimum value accepted is 20 (if you provide a value
# less than 20, it is reset to 0 and a warning is issued).  This limit is
# defined because otherwise you'll spend more time re-flushing a too small cache
# than working
# c.ZMQInteractiveShell.cache_size = 1000

# Automatically call the pdb debugger after every exception.
# c.ZMQInteractiveShell.pdb = False

#------------------------------------------------------------------------------
# ProfileDir configuration
#------------------------------------------------------------------------------

# An object to manage the profile directory and its resources.
# 
# The profile directory is used by all IPython applications, to manage
# configuration, logging and security.
# 
# This object knows how to find, create and manage these directories. This
# should be used by any code that wants to handle profiles.

# Set the profile location directly. This overrides the logic used by the
# `profile` option.
# c.ProfileDir.location = u''

#------------------------------------------------------------------------------
# Session configuration
#------------------------------------------------------------------------------

# Object for handling serialization and sending of messages.
# 
# The Session object handles building messages and sending them with ZMQ sockets
# or ZMQStream objects.  Objects can communicate with each other over the
# network via Session objects, and only need to work with the dict-based IPython
# message spec. The Session will handle serialization/deserialization, security,
# and metadata.
# 
# Sessions support configurable serialiization via packer/unpacker traits, and
# signing with HMAC digests via the key/keyfile traits.
# 
# Parameters ----------
# 
# debug : bool
#     whether to trigger extra debugging statements
# packer/unpacker : str : 'json', 'pickle' or import_string
#     importstrings for methods to serialize message parts.  If just
#     'json' or 'pickle', predefined JSON and pickle packers will be used.
#     Otherwise, the entire importstring must be used.
# 
#     The functions must accept at least valid JSON input, and output *bytes*.
# 
#     For example, to use msgpack:
#     packer = 'msgpack.packb', unpacker='msgpack.unpackb'
# pack/unpack : callables
#     You can also set the pack/unpack callables for serialization directly.
# session : bytes
#     the ID of this Session object.  The default is to generate a new UUID.
# username : unicode
#     username added to message headers.  The default is to ask the OS.
# key : bytes
#     The key used to initialize an HMAC signature.  If unset, messages
#     will not be signed or checked.
# keyfile : filepath
#     The file containing a key.  If this is set, `key` will be initialized
#     to the contents of the file.

# Username for the Session. Default is your system username.
# c.Session.username = 'fjd29'

# The name of the packer for serializing messages. Should be one of 'json',
# 'pickle', or an import name for a custom callable serializer.
# c.Session.packer = 'json'

# The UUID identifying this session.
# c.Session.session = u''

# execution key, for extra authentication.
# c.Session.key = ''

# Debug output in the Session
# c.Session.debug = False

# The name of the unpacker for unserializing messages. Only used with custom
# functions for `packer`.
# c.Session.unpacker = 'json'

# path to file containing execution key.
# c.Session.keyfile = ''

#------------------------------------------------------------------------------
# MappingKernelManager configuration
#------------------------------------------------------------------------------

# A KernelManager that handles notebok mapping and HTTP error handling

# The max raw message size accepted from the browser over a WebSocket
# connection.
# c.MappingKernelManager.max_msg_size = 65536

# Kernel heartbeat interval in seconds.
# c.MappingKernelManager.time_to_dead = 3.0

#------------------------------------------------------------------------------
# NotebookManager configuration
#------------------------------------------------------------------------------

# Automatically create a Python script when saving the notebook.
# 
# For easier use of import, %run and %loadpy across notebooks, a <notebook-
# name>.py script will be created next to any <notebook-name>.ipynb on each
# save.  This can also be set with the short `--script` flag.
# c.NotebookManager.save_script = False

# The directory to use for notebooks.
# c.NotebookManager.notebook_dir = '/home/fjd29'
