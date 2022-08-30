# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:13:14 2021

@author: Hideshi_Ooka
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg

config = {
    "font.size":20,
    "axes.linewidth":2.0,
    "xtick.top":True,
    "ytick.right":True,
    "xtick.direction":"in",
    "ytick.direction":"in",
    "xtick.major.size":8.0,
    "ytick.major.size":8.0,
    "xtick.major.width":2.0,
    "ytick.major.width":2.0,
    "legend.fontsize":8,
    "legend.frameon":False,
    "lines.markersize":5,
    "axes.labelpad":10,
    "font.family":"Helvetica"
}
plt.rcParams.update(config)
########################################
### UNIVERSAL LISTS AND DEFINITIONS ####
### THEY WILL BE IN CAPS            ####
########################################
LABEL_LIST = ["NO","N$_2$O","NH$_4^+$"]
C_LIST = ["b","r","g"]
MATERIAL_LIST = ["1T5.5","1T6.0"] 
PRODUCT_LIST = ["NO","N2O","NH4"]
pH_LIN = np.linspace(4,8,5)
E_LIN = np.linspace(0.2,-0.2,5)
pH, E = np.meshgrid(pH_LIN, E_LIN) # E is in RHE units
k_arr = np.zeros((25,3))
k_arr[:,0] = pH.flatten()
k_arr[:,1] = E.flatten()
LEN_EXPAND =101
pH_EXPAND_1D = np.linspace(4,8,LEN_EXPAND)
E_EXPAND_1D = np.linspace(0.2,-0.2,LEN_EXPAND)
pH_EXPAND_2D, E_EXPAND_2D = np.meshgrid(pH_EXPAND_1D, E_EXPAND_1D) 


#%%#################
### FUNCTIONS ######
####################
def split_SPET(logk1): 
    logk1 =pd.DataFrame(data = logk1, index =["0.2", "0.1","0","-0.1","-0.2"], columns =["4", "5", "6", "7","8"])
    apex = logk1.idxmax(axis=1)
    arr = logk1.values
    apex_idx = np.nanargmax(arr,axis=1)
    truth_array_ET = np.ones((5, 5), dtype=bool)
    truth_array_PT = np.ones((5, 5), dtype=bool)
    for idx in range(5):
        for jdx in range(5):
            if jdx > apex_idx[idx]:
                truth_array_ET[idx,jdx] = False
            if jdx < apex_idx[idx]:
                truth_array_PT[idx,jdx] = False
    logk1_ET = logk1.where(truth_array_ET)
    logk1_PT = logk1.where(truth_array_PT)
    return logk1_ET, logk1_PT

def get_coef(logk_E):
    k_arr[:,2] = logk_E.flatten()
    data = k_arr
    data = data[~np.isnan(data).any(axis=1)] # drops the rows with nan
    data = data[~np.isinf(data).any(axis=1)] # drops the rows with nan
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    coef,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    return coef   

def get_FE(v_obs_arr):
    C_vec = [1,4,6,1] # Coulombs
    C =np.zeros(v_obs_arr.shape)
    for idx in range(len(C_vec)):
        C[idx] = v_obs_arr[idx] * C_vec[idx]
    C_total = np.sum(C, axis = 0)
    FE = np.zeros(v_obs_arr.shape)
    for idx in range(len(C_vec)):
        FE[idx] = C[idx]/C_total * 100
    return FE

def get_logk(coef):
    if EXPAND == True:
        return coef[0]*pH_EXPAND_2D+coef[1]*E_EXPAND_2D + coef[2]
    else:
        return coef[0]*pH+coef[1]*E + coef[2]

def get_v_obs_arr(c_vec):
    # Under the assumption that NOads =1
    k0 = 10**get_logk(c_vec[0:3])
    k2 = 10**get_logk(c_vec[3:6])
    k3 = 10**get_logk(c_vec[6:9])
    k1_ET = 10**get_logk(c_vec[9:12])
    k1_PT = 10**get_logk(c_vec[12:15])
    k1 = np.minimum(k1_ET, k1_PT)
    if MECHANISM == "ER":
        v_obs_arr = np.array([k0/(k1+1),k0*k1/(k1+1),k2,k3])
    elif MECHANISM =="LH":
        v_obs_arr = np.array([k0,k1,k2,k3])
    return v_obs_arr

def get_k_E(jpartial_E_arr):
    # Under the assumption that NOads =1
    # The results can be used either directly or as seeds to let the NOads vary
    if MECHANISM == "ER": # Eley-Rideal
        k0 = jpartial_E_arr[0] + jpartial_E_arr[1] # NOsol is consumed by N2O formation
        k1 = jpartial_E_arr[1]/jpartial_E_arr[0] 
        k2 = jpartial_E_arr[2] 
        k3 = jpartial_E_arr[3] # this 3rd process is the residue    
    elif MECHANISM == "LH": # Langmuir-Hinshelwood   
        k0,k1,k2,k3 = jpartial_E_arr 
        # N2O_E = k1*NOads**2, this gives N2O_E = k1 when NOads is assumed to be 1
    return k0, k1, k2, k3

#%%#######################################
#### FUNCTIONS FOR OPTIMIZING c_vec ######
##########################################

def get_R2(c_vec):
    penalty_FE = get_penalty_FE(c_vec) 
    penalty_NOads = get_penalty_NOads(c_vec) 
    return penalty_FE # + 2000 * penalty_NOads

def get_penalty_FE(c_vec):
    v_obs_arr = get_v_obs_arr(c_vec)
    FE_T_arr = get_FE(v_obs_arr)
    if EXPAND == True:
        FE_T_arr = FE_T_arr[:,::int((len_expand-1)/4),::int((len_expand-1)/4)]
    delta_FE_arr = FE_T_arr-FE_E_arr
    penalty_FE = np.nansum(delta_FE_arr[0]**2 + delta_FE_arr[1]**2 + delta_FE_arr[2]**2)
    return penalty_FE
    
def get_penalty_NOads(c_vec):
    u0,u1,u2 = jpartial_E_arr[0:3]
    k0 = 10**get_logk(c_vec[0:3])
    k2 = 10**get_logk(c_vec[3:6])
    k3 = 10**get_logk(c_vec[6:9])
    k1_ET = 10**get_logk(c_vec[9:12])
    k1_PT = 10**get_logk(c_vec[12:15])
    k1 = np.minimum(k1_ET, k1_PT)
    
    if MECHANISM == "ER":
        NOads0 = (u0+u1)/k0
        NOads1 = u1/u0/k1
        NOads2 = u2/k2
    elif MECHANISM == "LH":
        NOads0 = u0/k0
        NOads1 = np.sqrt(u1/k1)
        NOads2 = u2/k2
    penalty_NOads = np.sum((NOads0-NOads1)**2 + (NOads1-NOads2)**2 + (NOads2-NOads0)**2)
    return penalty_NOads

def optimize(c_vec):
    L = len(c_vec)
    delta = 0.01
    alpha = 0.5
    R2 = get_R2(c_vec)
    R2_prev = R2
    print("Optimization for {}, {}:".format(material, MECHANISM))
    print("The R2 before GD is {}".format(R2))
    
    for jdx in range(1000):# do 10 times
        for idx in range(L):
            delta_vec = np.zeros(L)
            delta_vec[idx] = delta
            R2_plus = get_R2(c_vec + delta_vec)
            R2_neg = get_R2(c_vec - delta_vec)
            if R2_plus == np.min([R2, R2_plus, R2_neg]):
                c_vec = c_vec + delta_vec
                R2 = R2_plus
            elif R2_neg == np.min([R2, R2_plus, R2_neg]):
                c_vec = c_vec - delta_vec
                R2 = R2_neg
            else:
                pass
        if jdx % 50 == 0:
            print("jdx = {}, current R2 = {}".format(jdx, R2))                    
        if R2_prev == R2:
            print("jdx = {}, current R2 = {}".format(jdx, R2))           
            print("========================================")
            print(c_vec)
            print(" ")
            print(" ")
            print(" ")
            break
        R2_prev = R2
    return c_vec

def make_csv_of_E(material):
    # Raw data is "pH-Eh-activity.xlsx, obtained on Jan 21 20:37, 2021, reorganized into a CSV of each individual product
    jpartial_E_list = [material+"_"+product+"_jpartial_E.csv" for product in PRODUCT_LIST]
    FE_E_list = [material+"_"+product+"_FE_E.csv" for product in PRODUCT_LIST]

    FE_E_arr = np.zeros((4,5,5))
    for i,file in enumerate(FE_E_list):
        df = pd.read_csv(file, index_col = 0)
        FE_E_arr[i] = df.values
    FE_E_arr[3] = 100-np.sum(FE_E_arr[0:3], axis = 0)
    
    jpartial_E_arr = np.zeros((4,5,5))
    for i,file in enumerate(jpartial_E_list):
        df = pd.read_csv(file, index_col = 0)
        jpartial_E_arr[i] = df.values
    jpartial_E_arr[3] = np.sum(jpartial_E_arr[0:3], axis = 0) * FE_E_arr[3]/ np.sum(FE_E_arr[0:3],axis=0)
    
    np.savetxt("{}_FE_E.csv".format(material), FE_E_arr.reshape(20,5), delimiter = ",")
    np.savetxt("{}_jpartial_E.csv".format(material), jpartial_E_arr.reshape(20,5), delimiter = ",")

#%%############################################
######## NUMERICAL ANALYSIS ###################
###############################################
MECHANISM = "LH" 
# choose ER (Eley-Rideal) or LH (Langmuir-Hinshelwood)
# both mechanisms (ER and LH) seem to be able to reproduce experimental trends, but isotope experiments suggest ER

EXPAND = False
for material in MATERIAL_LIST:
    # make_csv_of_E(material) # This is used only once
    FE_E_arr = np.loadtxt("{}_FE_E.csv".format(material), delimiter=",").reshape(4,5,5)
    jpartial_E_arr = np.loadtxt("{}_jpartial_E.csv".format(material), delimiter=",").reshape(4,5,5)
    k0, k1, k2, k3 = get_k_E(jpartial_E_arr) 
    # get_k_E() assumes coverage of NOads is 1.
    # the results can be used directly, or as a seed for further optimization
    k1_ET, k1_PT = split_SPET(k1)
    k_E_arr = np.array([k0,k1,k2, k3, k1_ET,k1_PT])
    
    logk_E_arr = np.log10(k_E_arr)  
    logk_T_arr = np.zeros(logk_E_arr.shape)
    c_vec = np.zeros(3*len(logk_T_arr)) # 3 parameters for each plane
    for i,logk_E in enumerate(logk_E_arr):
        coef = get_coef(logk_E)
        logk_T = get_logk(coef)# T stands for theory 
        logk_T_arr[i] = logk_T
        c_vec[3*i:3*(i+1)]=coef
    logk_T_arr[1] = np.minimum(logk_T_arr[-2],logk_T_arr[-1]) 
    c_vec_initial = np.delete(c_vec, [3,4,5])
   
    ##############################################
    ### OPTIMIZE THE c_vec #######################
    ##############################################
    c_vec_opt = c_vec_initial # optimize(c_vec_initial)
    
    
    ##############################################
    ### CALCULATE THE FE #########################
    ##############################################    
    
    v_obs_arr = get_v_obs_arr(c_vec_opt)
    FE_T_arr = get_FE(v_obs_arr)    
    
    #heatmap2d(v_obs_arr[0])
    #heatmap2d(jpartial_E_arr[0])
    
    np.savetxt("{}_c_vec_initial_{}.csv".format(material, MECHANISM), c_vec_initial.reshape(5,3), delimiter = ",")
    np.savetxt("{}_c_vec_optimized_{}.csv".format(material, MECHANISM), c_vec_opt.reshape(5,3), delimiter = ",")
    np.savetxt("{}_FE_T_{}.csv".format(material, MECHANISM), FE_T_arr.reshape(20,5), delimiter =",")
    
#%%############################################
######## MAKE THE 3D FIGURE ###################
###############################################
EXPAND = True # expand the number of theoretical datapoints

fig=plt.figure(figsize = (15,12))
for i in range(6):
    ax = fig.add_subplot(2,3,i+1, projection = "3d")
    ax.set_xlabel("\npH")            
    ax.set_ylabel("\n$E$ (V vs. RHE)")            
    ax.set_zlabel("\n$j_{partial}$ (%)")           
    ax.set_xticks(np.linspace(4,8,3))
    ax.set_yticks(np.linspace(-0.2,0.2,3))
    j = i%3
    
    if i ==0:
        jpartial_E_arr = np.loadtxt("1T6.0_jpartial_E.csv", delimiter=",").reshape(4,5,5)
        c_vec_60 = np.loadtxt("1T6.0_c_vec_optimized_{}.csv".format(MECHANISM), delimiter = ",").flatten()
        v_obs_arr = get_v_obs_arr(c_vec_60)
    if i == 3:
        jpartial_E_arr = np.loadtxt("1T5.5_jpartial_E.csv", delimiter=",").reshape(4,5,5)
        c_vec_55 = np.loadtxt("1T5.5_c_vec_optimized_{}.csv".format(MECHANISM), delimiter = ",").flatten()        
        v_obs_arr = get_v_obs_arr(c_vec_55)
    ax.scatter(pH, E, jpartial_E_arr[j], color=C_LIST[j])
    ax.plot_surface(pH_EXPAND_2D, E_EXPAND_2D, v_obs_arr[j], color = C_LIST[j], alpha = 0.2)
    ax.set_zlim(0,500)
    # Plot the error bars
    xT = pH_EXPAND_2D[::int((LEN_EXPAND-1)/4),::int((LEN_EXPAND-1)/4)]
    yT = E_EXPAND_2D[::int((LEN_EXPAND-1)/4),::int((LEN_EXPAND-1)/4)]
    zT = v_obs_arr[j][::int((LEN_EXPAND-1)/4),::int((LEN_EXPAND-1)/4)]
    for k in range(25):
        x = [pH.flatten()[k],xT.flatten()[k]]
        y = [E.flatten()[k],yT.flatten()[k]]
        z = [jpartial_E_arr[j].flatten()[k],zT.flatten()[k]]
        ax.plot(x,y,z,c = C_LIST[j], lw = 2, ls = "-")
    plt.subplots_adjust(wspace  =0.5, hspace = 0.2)
plt.title(str(MECHANISM))
plt.savefig("jpartial_{}.png".format(MECHANISM), dpi = 300, transparent = True)  



#%%
fig=plt.figure(figsize = (15,12))
for i in range(6):
    ax = fig.add_subplot(2,3,i+1, projection = "3d")
    ax.set_xlabel("\npH")            
    ax.set_ylabel("\n$E$ (V vs. RHE)")            
    ax.set_zlabel("\nFE (%)")           
    ax.set_xticks(np.linspace(4,8,3))
    ax.set_yticks(np.linspace(-0.2,0.2,3))
    j = i%3
    
    if i ==0:
        FE_E_arr = np.loadtxt("1T6.0_FE_E.csv", delimiter=",").reshape(4,5,5)
        c_vec_60 = np.loadtxt("1T6.0_c_vec_optimized_{}.csv".format(MECHANISM), delimiter = ",").flatten()
        FE_T_arr = get_FE(get_v_obs_arr(c_vec_60))
    if i == 3:
        FE_E_arr = np.loadtxt("1T5.5_FE_E.csv", delimiter=",").reshape(4,5,5)
        c_vec_55 = np.loadtxt("1T5.5_c_vec_optimized_{}.csv".format(MECHANISM), delimiter = ",").flatten()        
        FE_T_arr = get_FE(get_v_obs_arr(c_vec_55))
    ax.scatter(pH, E, FE_E_arr[j], color=C_LIST[j])
    ax.plot_surface(pH_EXPAND_2D, E_EXPAND_2D, FE_T_arr[j], color = C_LIST[j], alpha = 0.2)
    ax.set_zlim(0,100)
    # Plot the error bars
    xT = pH_EXPAND_2D[::int((LEN_EXPAND-1)/4),::int((LEN_EXPAND-1)/4)]
    yT = E_EXPAND_2D[::int((LEN_EXPAND-1)/4),::int((LEN_EXPAND-1)/4)]
    zT = FE_T_arr[j][::int((LEN_EXPAND-1)/4),::int((LEN_EXPAND-1)/4)]
    for k in range(25):
        x = [pH.flatten()[k],xT.flatten()[k]]
        y = [E.flatten()[k],yT.flatten()[k]]
        z = [FE_E_arr[j].flatten()[k],zT.flatten()[k]]
        ax.plot(x,y,z,c = C_LIST[j], lw = 2, ls = "-")
    plt.subplots_adjust(wspace  =0.5, hspace = 0.2)
plt.savefig("FE_{}.png".format(MECHANISM), dpi = 300, transparent = True)  

