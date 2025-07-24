
import numpy as np
import pandas as pd

##########################
### Physical Constants ###
##########################
AREA = np.pi * 0.55*0.55/4
FRT = 96485/8.314/300
nFN = 1

#############################################
############ IMPORT DATA ####################
#############################################
def read_CV(filename = "Pt_HER_1M.csv"):
    df = pd.read_csv(filename)
    all_E = df.dropna()["Ecorr_3600"].values # units: V vs. RHE after IR correction
    all_j_exp = df.dropna()["i_3600"].values/AREA # units: mA/cm2 geometric area of RDE 
    j_exp = all_j_exp[all_E<-0.01] # choose region for fitting
    E = all_E[all_E < -0.01] # choose region for fitting
    log_j_exp = np.log10(np.abs(j_exp))
    all_log_j_exp = np.log10(np.abs(all_j_exp))
    return all_E, all_j_exp, all_log_j_exp, E, log_j_exp
all_E, all_j_exp, all_log_j_exp, E, log_j_exp = read_CV()


def get_log_j_VHT(par_vec, mechanism = "VHT", const_kzeros = False, alpha_lock = 0, return_contribution = False, E=E):
    """
    The par_vec to be inputted:
    GH,logK2,logK3,a1, a2, a3 = par_vec
    GH: Binding Energy
    a1 to a3: Electron transfer coefficient/BEP coefficient of each elementary step
    """
    GH,logK2,logK3,a1, a2, a3 = par_vec
    K2 = 10**logK2
    K3 = 10**logK3
    if mechanism == "VT": # 
        K2 = 0
    if mechanism == "VH":
        K3 = 0
    k10 = 1
    k20 = k10 * K2
    k30 = k10 * K3    
    if const_kzeros == True:
        k20 = np.sqrt(K2)
        k30 = np.sqrt(K3)
        if K2 < K3:
            k10 = 1/np.sqrt(K3)
        else:
            k10 = 1/np.sqrt(K2)    
    if alpha_lock != 0:
        a1 = alpha_lock
        a2 = alpha_lock
        a3 = alpha_lock
    Eads = -GH # expressing GH in eV units would allow the Faraday constant to be ommitted
    # X [eV] x NA (Avogadro) = X[V] * 1.602 x 10-19 [C] x NA = X[V] x Faraday constant
    a1r = 1-a1
    a2r = 1-a2
    a3r = 1-a3
    k1  = k10 * np.exp(a1 *FRT*( Eads-E))        
    k1r = k10 * np.exp(a1r*FRT*(-Eads+E))        
    k2  = k20 * np.exp(a2 *FRT*(-Eads-E))
    k2r = k20 * np.exp(a2r*FRT*( Eads+E))
    k3  = k30 * np.exp(a3 *FRT*(-2*Eads))
    k3r = k30 * np.exp(a3r*FRT*  2*Eads )
    gamma = np.sqrt( (k1+k1r+k2+k2r)**2 + 4*(k3*(k1+k2r)+k3r*(k1r+k2)+k3*k3r) )
    theta = 2*(k1+k2r+k3r)/(k1+k1r+k2+k2r+2*k3r+gamma)
    v1 = k1*(1-theta) - k1r*theta
    v2 = k2*(theta) - k2r*(1-theta)
    v3 = k3*(theta)**2 - k3r*(1-theta)**2
    j_VHT = 2*nFN*(k1**2*(k2+k3)-k1r**2*(k2r+k3r) +k1*k1r*(k2-k2r))/((k1+k1r)*(k1+k1r+k2+k2r+gamma)+2*(k1*k3+k1r*k3r))
    if return_contribution == True:
        return np.log10(np.abs(j_VHT)), theta, np.log10(v1),np.log10(v2),np.log10(v3)
    else:
        return np.log10(np.abs(j_VHT))  

def get_offset(log_j_VHT):
    offset = np.average(log_j_VHT - log_j_exp)
    return offset

def get_offset_log_j_VHT(log_j_VHT):
    return log_j_VHT - get_offset(log_j_VHT)

def get_diff(par_vec):
    log_j_VHT = get_log_j_VHT(par_vec)
    diff = np.average((get_offset_log_j_VHT(log_j_VHT) - log_j_exp)**2)
    return diff

def get_log_diff(par_vec):
    diff = get_diff(par_vec)
    return np.log10(diff)
    
