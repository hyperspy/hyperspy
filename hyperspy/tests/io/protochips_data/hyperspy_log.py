#!/usr/bin/env python 
# ============================
# 2016-07-17 
# 01:27 
# ============================
get_ipython().magic('ls ')
s = hs.load('aduro_log_file.csv')
s = hs.load('protochips_electrical.csv')
s.plot()
s
hs.plot.plot_spectra(s[:2])
