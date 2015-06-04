c = get_config()
c.IPKernelApp.exec_lines = ['import hyperspy.hspy as hs',
                            'import hyperspy.Release', ]
c.IPKernelApp.pylab = "qt"
