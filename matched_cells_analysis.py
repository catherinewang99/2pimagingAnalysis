#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:42:58 2023

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure
import decon
from scipy.stats import chisquare
import pandas as pd


path = r'F:\data\BAYLORCW036\python\2023_10_07'
s1 = session.Session(path)

path = r'F:\data\BAYLORCW036\python\2023_10_09'
s2 = session.Session(path)



