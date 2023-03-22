# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:03:02 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure

### Comparison of epoch specific selective cells

g_res = []
m_res = []

g_del = []
m_del = []

g_sam = []
m_sam = []

for i in range(5):
    
    path = r'F:\data\BAYLORCW021\python\2023_03_20'
    l1 = session.Session(path, i+1)
    
    path = r'F:\data\GC225\python\2022_02_14'
    gc = session.Session(path, i+1, guang = True)
    print(" --- LAYER {} ---".format(i + 1))
    print("Guang's proportion of response selective neurons: {}".format(len(gc.get_response_selective()) / gc.num_neurons))
          
    print("My proportion of response selective neurons: {}".format(len(l1.get_response_selective()) / l1.num_neurons))
    
    
    g_res += [len(gc.get_response_selective()) / gc.num_neurons]
    m_res += [len(l1.get_response_selective()) / l1.num_neurons]
    
    g_del += [len(gc.get_delay_selective()) / gc.num_neurons]
    m_del += [len(l1.get_delay_selective()) / l1.num_neurons]
    
    g_sam += [len(gc.get_sample_selective()) / gc.num_neurons]
    m_sam += [len(l1.get_sample_selective()) / l1.num_neurons]
    

f, axarr = plt.subplots(3,1)
axarr[0].plot(g_res, 'g-')
axarr[0].plot(m_res, 'g--')
axarr[1].plot(g_del, 'r-')
axarr[1].plot(m_del, 'r--')
axarr[2].plot(g_sam, 'b-')
axarr[2].plot(m_sam, 'b--')



plt.plot(g_res, 'g-')
plt.plot(m_res, 'g--')
plt.plot(g_del, 'r-')
plt.plot(m_del, 'r--')
plt.plot(g_sam, 'b-')
plt.plot(m_sam, 'b--')


### Distribution of selectivity within response-selective cells

path = r'F:\data\BAYLORCW021\python\2023_02_15'
l1 = session.Session(path, 6)

path = r'F:\data\GC225\python\2022_02_14'
gc = session.Session(path, 6, guang = True)

gc_sel = gc.get_response_selective()
my_sel = l1.get_response_selective()

gc_hist = list()
my_hist = list()

for n in gc_sel:
    gc_hist.append(np.median(gc.plot_selectivity(n, plot=False)))
for n in my_sel:
    my_hist.append(np.median(l1.plot_selectivity(n, plot=False)))
    
plt.hist(gc_hist, bins='auto', alpha=0.5, color='b')
plt.hist(my_hist, bins='auto', alpha=0.5, color='g')

    




























