#%%# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 20:08:44 2022
@author: Hideshi_Ooka
"""
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit # used to fit MM
from sklearn.linear_model import LinearRegression # used to fit Km-kcat scaling
R = 8.314 # J/mol K
PSP_types = pd.read_csv("PSP_types.csv", index_col = 0)
ORG_LIST_FULL = PSP_types.index.tolist()
N = len(ORG_LIST_FULL)

#######################
### Fit Km and kcat ### 
#######################
def MM(S,Km,k2):
    return k2*S/(Km+S)
def get_MM_pars(df, keep_highest_conc = True):
    S = df.columns.values.astype(float)
    N = df.shape[0]
    data = df.values
    Km_arr=np.array([])
    kcat_arr=np.array([])
    if keep_highest_conc==False: # skip S=10 for PSP data due to possibility of S inhibition
        S = S[:-1]
        data = data[:,:-1]
    for i in range(N):
        v = data[i]
        if v.max() > 0:
            popt, pcov= curve_fit(MM,S,v,p0=[0.01,1])    
            Km,kcat = popt
        else:
            Km,kcat = np.nan, np.nan
        Km_arr = np.append(Km_arr,Km)
        kcat_arr = np.append(kcat_arr,kcat)
    return Km_arr, kcat_arr
#####################
### Fit Arrhenius ### 
#####################
def fit_Arrhenius(lnk2, T_arr):
    Y = lnk2.flatten()
    valid_indices = ~np.isnan(Y)
    Y = Y[valid_indices]
    X = np.ones((len(Y),2))
    T_arr = T_arr[valid_indices]
    X[:,0] = 1/T_arr
    a,b = np.linalg.inv(X.T@X)@(X.T@Y)# y = ax + b
    lnA = b
    A = np.exp(lnA)
    Ea = -a*R # J/mol units
    return lnA, A, Ea

def save_Arrhenius_pars(lnk2_file):
    lnk2 = pd.read_csv(lnk2_file, index_col = 0)
    T_arr = lnk2.columns.astype(float).to_numpy()
    T_arr += 273.15 # convert to kelvin
    lnA2_arr =np.zeros(N)
    A2_arr =np.zeros(N)
    Ea2_arr =np.zeros(N)
    for i in range(N):
        lnA2, A2, Ea2 = fit_Arrhenius(lnk2.values[i], T_arr)
        lnA2_arr[i] = lnA2
        A2_arr[i] = A2
        Ea2_arr[i] = Ea2
    df = pd.DataFrame([lnA2_arr,A2_arr,Ea2_arr]).T
    df.columns=["lnA2","A2","Ea2"]
    df.index=ORG_LIST_FULL
    df.to_csv("Analyzed/PSP_Arrhenius_pars.csv")

################################
### Fit Scaling Relationship ### 
################################
    
def get_y_pred(x,y):
    reg = LinearRegression()
    reg.fit(x,y)
    y_pred = reg.predict(x) 
    return y_pred

def get_r2(y, y_pred):
    y_avg = np.mean(y)
    r2 = 1 - np.sum((y-y_pred)**2)/np.sum((y - y_avg)**2)
    return r2

def get_alpha_C(Km_arr,kcat_arr, alpha):
    valid_indices = ~np.isnan(Km_arr)
    x = np.log10(Km_arr[valid_indices])
    y = np.log10(kcat_arr[valid_indices])
    if alpha == 0: # unconstrained
        x = x.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(x,y)
        x = x.flatten()
        alpha = reg.coef_[0]

    intercepts = np.zeros(len(valid_indices))    
    intercepts[valid_indices] = y-alpha*x
    logC_fit = np.mean(intercepts)
    logC = intercepts
    return alpha, logC, logC_fit

def save_pars(v_avg_file, alpha, keep_highest_conc = True):
    v = pd.read_csv(v_avg_file, index_col = 0)
    Km_arr, kcat_arr = get_MM_pars(v, keep_highest_conc)
    alpha, logC, logC_fit = get_alpha_C(Km_arr, kcat_arr, alpha)
    pars = np.zeros((v.shape[0], 7))
    pars[:,0] = Km_arr
    pars[:,1] = kcat_arr
    pars[:,2] = alpha
    pars[:,3] = logC
    pars[:,4] = logC_fit
    pars[:,5] = 10**(logC)
    pars[:,6] = 10**(logC_fit)
    df = pd.DataFrame(pars)
    df.index = v.index
    df.columns = ["Km","kcat","alpha","logC","logC_fit","C","C_fit"]
    par_file_name = v_avg_file.replace("v_avg.csv",f"pars_{str(alpha)[:3]}.csv")
    df.to_csv(par_file_name)
    round_file_name = par_file_name.replace(".csv","_Round.csv")
    df.to_csv(round_file_name, float_format="%.3g")

if __name__ == "__main__":
    for T in [40,70]:
        for alpha in [0,0.2,0.5,0.8]:
            save_pars(f"Analyzed/PSP_{T}deg_v_avg.csv", alpha, keep_highest_conc=False)
    save_pars("Analyzed/Cellulases_v_avg.csv", 0, keep_highest_conc=True)
    save_Arrhenius_pars("Analyzed/PSP_lnk2_avg.csv")