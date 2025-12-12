#%%############################################################################
### This code just performs calculations and saves the results as csv files ###
### Run figures.py afterwards to generate figures #############################
###############################################################################
import numpy as np
import pandas as pd
from GA_utils import save_pars
pop_list = [10,30,50,100,300,500,1000]
# Perform GA and Save the parameters
for pop_size in pop_list:
    save_pars(pop_size)
#%% Calculate the statistics
num_pars = 6 # this is dependent on what to fit but still hardcoded
M_Mean = np.zeros((len(pop_list), num_pars+2))
M_STD =  np.zeros((len(pop_list), num_pars+2))
for alpha in [0, 0.5]:
    for i, pop in enumerate(pop_list):
        df = pd.read_csv(f"fitting_results/results_table_VHT_{pop}_{alpha}.csv")
        M_Mean[i,1:] = df.mean()
        M_STD[i,1:] = df.std()
    M_Mean[:,0] = pop_list
    M_STD[:,0] = pop_list
    np.savetxt(f"M_Mean_VHT_{alpha}.csv", M_Mean, delimiter=',', header="pop,GH, logK2, logK3, a1,a2,a3,log_diff")      
    np.savetxt(f"M_STD_VHT_{alpha}.csv", M_STD, delimiter=',', header="pop,GH, logK2, logK3, a1,a2,a3,log_diff") 
par_vec = df.mean()[:-1]