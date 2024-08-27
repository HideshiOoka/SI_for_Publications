#%%
import pandas as pd
import numpy as np
import os    

def get_tau(CA_file, evaluation_percentages):
    t,j = pd.read_csv(insitu_dir + CA_file).values.T
    N = len(t) # number of datapoints
    j0 = j[:N//10].mean()
    tau_list =[]
    for evaluation_percentage in evaluation_percentages:
        tau = t[np.abs(j - j0*evaluation_percentage/100).argmin()]
        tau_list.append(tau)
    return tau_list,j0

def get_vd(fitted_file):
    fitted =pd.read_csv(insitu_dir + fitted_file).values
    t,III,VII,baseline,III_Pi,r2 = fitted.T
    III+= III_Pi
    g_list = [] # gradient
    avg_g_list = []
    for i in range(len(t)-1):
        g = (VII[i+1]-VII[i])/(t[i+1]-t[i])
        g_list.append(g)
        if i > 10: # start to check the gradient
            recent_g = np.array(g_list[-10:])
            ratio = np.median(recent_g)/recent_g.mean()
            if ratio < 1.1 and ratio > 0.9:
                vd = recent_g.mean()
                break
    return vd
insitu_dir = "Experimental_Data/insitu_UVVis/"
fitted_list = [f for f in os.listdir(insitu_dir) if f[-10:]=="Fitted.csv"] 
evaluation_percentages = [10,37, 50, 90] # 100/e = 36.787944117144235
summary =[]
N = len(fitted_list)
M = len(evaluation_percentages) + 3

for fitted_file in fitted_list:
    date, ID = fitted_file.split("_")[:2]
    CA_file = fitted_file.replace("Fitted","CA")
    tau_list,jOER = get_tau(CA_file, evaluation_percentages) # this returns jOER at initial time (10% of CA measurement)
    vd = get_vd(fitted_file)
    summary += [ID,vd,jOER] + tau_list
summary_arr = np.reshape(summary, (N,M))      
column_list =  ["ID","vd","jOER"] + [f"tau{percentage}" for percentage in evaluation_percentages]
df = pd.DataFrame(summary_arr, columns =column_list)
df.set_index("ID")
df.to_csv("Experimental_Data/MnO2.csv", index = False)

