#%% -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:12:09 2022
@author: Hideshi_Ooka
This code makes the X matrix for fitting the UV-Vis spectra.
The X matrix will be used to fit the spectra in the insitu_UVVis directory based on the equation
XT y = XTX beta
"""
import numpy as np
import pandas as pd
from numpy.linalg import inv 
import os

#################################################################
### Reference spectra are read here. ############################ 
### This is used to make the X matrix for UV-Vis fitting ########
#################################################################
ref_dir = "Experimental_Data/Reference_Spectra/"
analysis_dir ="Analysis/"
def get_VII_ref(file, L = 1):
    df = pd.read_csv(ref_dir+file, index_col=0)
    conc = df.columns.astype(float)*1E-6 # conversion of microM to M units
    ref_spec = (df/conc).iloc[:,-1] / L
    epsilon = (df/conc).loc[545].iloc[1:] / L # optical path 1 cm
    epsilon = np.mean(epsilon)
    """
    print(epsilon) 
    This value returns 2035, which is a bit low compared to literature values of 2300. 
    This leads to underestimating the MnVII concentration and kd, but the deviation would be the same for every measurement.
    Therefore, this should not change qualitative trends like the linearity in the experimental/theoretical comparison.
    """
    return ref_spec

def get_III_ref(file, L = 1):
    df = pd.read_csv(ref_dir+file, index_col=0,skiprows=1) 
    # Get the last spectra in the time series
    conc = 100*1E-6 # Final concentration of Mn3+ in microM units, expected from the stoichiometry of the Guyard reaction: Mn7+ + 4Mn2+ -->5 Mn3+
    ref_spec = (df/conc).iloc[:,-1] / L
    return ref_spec

all_X = np.zeros((601,4))
all_X[:,0] = get_III_ref("210927_20uMMn7_200uMMn2_in1MH2SO4+2MNa2SO4.csv") # III in H2SO4
all_X[:,1] = get_VII_ref("Ref_Spec_Mn7_in1MH2SO42M+Na2SO4.csv") # VII in H2SO4
all_X[:,2] = np.ones(601) # A flat baseline, presumably due to bubbles
all_X[:,3] = get_III_ref("210924_20uMMn7_200uMMn2_in1MH3PO4.csv")
"""
The reference of VII in H3PO4 overlaps with the one in H2SO4 and was exluded from the fitting procedure.
See: Ref_Spec_Mn7_in1MH3PO4.csv
"""
#%%#####################################
### The actual fitting is done here ####
########################################
def get_coef(y, N, wavelength, fitting_start = 300, fitting_end = 800):
    start_ind = np.where(wavelength == fitting_start)[0][0]
    end_ind = np.where(wavelength == fitting_end)[0][0]+1
    full_X = all_X[:,:N]
    X = full_X[start_ind:end_ind]
    y = y[start_ind:end_ind]
    XTXinv =  inv(np.matmul(X.T, X))
    beta = np.matmul(XTXinv, np.dot(X.T,y))
    yhat = np.matmul(X,beta)
    full_yhat = np.matmul(full_X,beta)
    return beta, yhat,full_yhat

def save_fitted_UV_file(UV_file):
    fitted_file = UV_file.replace("UV","Fitted")
    ID = UV_file.split("_")[1]
    df = pd.read_csv(insitu_dir + UV_file, index_col = 0)
    df.columns = df.columns.astype(float)
    if df.shape[0] == 1201: # save the original file with 0.5 nm resolution, but analyze the spectra with 1 nm resolution
        df.to_csv(insitu_dir+UV_file.replace(".csv","_05nm_resolution.csv"))
        df = df.iloc[::2] # every 0.5 nm --> every 1 nm
        df.to_csv(insitu_dir+UV_file)
    y = df.values
    wavelength = np.linspace(200,800,601)
    results_arr = np.zeros((df.shape[1],6))
    results_arr[:,0] = df.columns
    conditions = summary[summary.ID == ID].conditions.values[0]
    N = 3
    if "+H3PO4" in conditions:
        N = 4 # also use the Mn3+ in Pi spectrum to fit
    beta, yhat,full_yhat = get_coef(y, N, wavelength)
    results_arr[:,1:N+1] = beta.T
    results_arr[:,5] = np.sum((full_yhat-y)**2, axis=0)
    results_df = pd.DataFrame(results_arr[:,1:], columns = ["III_H2SO4","VII","Baseline","III_H3PO4","r2"], index = df.columns)
    results_df.to_csv(insitu_dir+fitted_file)

insitu_dir = "Experimental_Data/insitu_UVVis/"
insitu_files = os.listdir(insitu_dir)
UV_files = [f for f in insitu_files if "UV.csv" in f]
summary = pd.read_csv("Experimental_Data/Summary_Experimental_Conditions.csv") # This is the index file
for UV_file in UV_files:
    save_fitted_UV_file(UV_file)
