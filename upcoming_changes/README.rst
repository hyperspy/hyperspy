This directory contains "news fragments" which are short files that contain a small **ReST**-formatted
text that will be added to the next ``CHANGELOG``.

The ``CHANGELOG`` will be read by **users**, so this description should be aimed to hyperspy users
instead of describing internal changes which are only relevant to the developers.

Each file should be named like ``<ISSUE>.<TYPE>.rst``, where
``<ISSUE>`` is an issue number, and ``<TYPE>`` is one of:

* ``new``: new user facing features, like new command-line options and new behavior.
* ``bugfix``: fixes a bug.
* ``doc``: documentation improvement, like rewording an entire session or adding missing docs.
* ``deprecation``: feature deprecation.
* ``enhancements``: improvement of existing functionality, usually without requiring user intervention.
* ``api``: a change which may break an existing script, such as feature removal or behavior change.
* ``maintenance``: a change related to the test suite, packaging, etc.

So for example ``1412.new.rst`` or ``2773.bugfix.rst``.

If your PR fixes an issue, use the number of the issue here. If there is no issue,
then after you submit the PR and get the PR number you can add a
changelog using that instead.

If you are not sure what issue type to use, don't hesitate to ask in your PR.

``towncrier`` preserves multiple paragraphs and formatting (code blocks, lists, and so on), but for entries
other than ``new`` it is usually better to stick to a single paragraph to keep it concise. For ``new``,
it is recommended to add a hyperlink to the user guide.

To make a draft of the changelog, run from the command line:

   .. code-block:: bash

       $ towncrier build --draft

See https://github.com/twisted/towncrier for more details.
