c = get_config()
c.IPKernelApp.exec_lines = ['import hyperspy.hspy as hs',
                            'import hyperspy.Release',
                            'print hyperspy.Release.info']
c.IPKernelApp.pylab = "qt"
