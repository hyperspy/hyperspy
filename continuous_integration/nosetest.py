"""Run nosetests after setting ETS toolkit to "null
"
"""
#!/usr/bin/python
import nose
from traits.etsconfig.api import ETSConfig


ETSConfig.toolkit = "null"
nose.run()
