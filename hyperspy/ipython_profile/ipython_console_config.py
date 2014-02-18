c = get_config()
c.IPKernelApp.exec_lines = ['from hyperspy.hspy import *',
                            'import hyperspy.Release', ]
c.IPKernelApp.pylab = "qt"
