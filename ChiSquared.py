#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:36:09 2023

@author: williamfloyd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:41:13 2023

@author: williamfloyd
%matplotlib inline 
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd


def draw_from_normal(df,sigma = 1,mean = 0):
    ans = 0
    for i in range(df):
        ans += pow(np.random.normal(mean,sigma),2)
    return ans


def simulate_sums(n,df,sigma=1,mean=0):
    ans = []
    for i in range(n):
        ans.append(draw_from_normal(df,sigma,mean))
    return ans


def plot_hist(data):
    upper_limit = max(data)
    lower_limit = min(data)
    num_bins = (upper_limit-lower_limit) // 10
    num_bins = 10
    num_bins += 1
    if num_bins < 20:
        num_bins = 20
    print(num_bins)
    #plt.hist(data,bins = int(num_bins),histtype = 'step',density=True)

num_sims = 100000

one_deg = simulate_sums(num_sims,1,1,0)
two_degs = simulate_sums(num_sims,2)
three_degs = simulate_sums(num_sims,3)
five_degs = simulate_sums(num_sims,5)
ten_degs = simulate_sums(num_sims,10)

my_data = {'Two degs':two_degs
           ,'Three degs':three_degs,'Five Degs':five_degs,
           'Ten Degs': ten_degs}

my_plot_data = pd.DataFrame(my_data)
#my_plot = pd.DataFrame([one_deg,two_degs],columns=['One Deg','Two Degs'])





#sns.displot(my_plot,x="Observed",kind='kde',bw_adjust=2,cut=0)
my_graph = sns.displot(data=my_plot_data,kind='kde')


plt.show()
#plot_hist(my_plot)

""