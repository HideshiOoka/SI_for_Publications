#%%
import numpy as np
from scipy.optimize import fsolve
RT = 8.314/1000*300
ET =0.01

def f(g1, *args):
    gT,a1,a2,k10,k20,S = args
    return (1-a2)/a2*g1 + a1/a2*k20/k10*g1**(a1+a2)/gT**a2-S

def get_dG1_from_Km(Km, dGT, k10=1,k20=1):
    gT = np.exp(dGT/RT) # dGT is defined in Fig. 2
    sgT = gT**0.5
    K = k20/k10/sgT
    g1 = Km/(1+K)
    dG1 = RT* np.log(g1)
    return dG1


def get_Km_v(dG1,dGT,S,k10=1,k20=1, a1 = 0.5, a2 = 0.5):
    g1 = np.exp(dG1/RT)
    gT = np.exp(dGT/RT)
    K = k20/k10/gT**a2   *g1**(a1+a2-1)
    Km = g1*(1+K)
    v = k20*g1**a2/gT**a2*S*ET/(S+Km)
    # "estimate" are the results obtained assuming a1 = a2 = 0.5
    estimate_opt_Km = S # by definition
    if gT.shape == ():
        estimate_K = k20/k10/gT**0.5
    else:
        estimate_K = k20/k10/gT[:,0]**0.5
    estimate_opt_g1 = S/(1+estimate_K)
    estimate_opt_dG1 = RT * np.log(estimate_opt_g1)
    # "true" are the results obtained under general a1 and a2
    if a1 == 0.5 and a2 == 0.5: # by definition, estimate = true
        true_opt_Km = estimate_opt_Km
        true_opt_dG1 = estimate_opt_dG1
    else: 
        # otherwise, numerical optimization is necessary to obtain the true optimum dG1 and optimum Km,
        # because K in the right hand side of Eq. 14 also depends on dG1
        true_opt_g1 = np.ones(g1.shape[0]) 
        for i, gT_val in enumerate(gT[:,0]):
            true_opt_g1[i]= fsolve(f, estimate_opt_g1[i], args = (gT_val,a1,a2,k10,k20,S))
        true_opt_Km = true_opt_g1*(1+K) 
        true_opt_dG1 = RT*np.log(true_opt_g1)
    return Km, v, estimate_opt_Km, true_opt_Km, estimate_opt_dG1, true_opt_dG1

def get_Km_v_reverse(dG1,dGT,S,P=0,k10=1,k20=1):
    g1 = np.exp(dG1/RT)
    gT = np.exp(dGT/RT)
    sg1 = np.sqrt(g1)
    sgT = np.sqrt(gT)
    K = k20/k10/sgT
    Km = g1*(1+K)
    v = k10*K*sg1*(S-gT*P)*ET/(Km + S+k20/k10*sgT*P)
    v[v<0]=1E-7
    estimate_opt_Km = S
    true_opt_Km = S +K*gT*P
    estimate_opt_dG1 = RT*(np.log(S) - np.log(1+K))[:,0]
    true_opt_g1 = true_opt_Km/(1+K)
    true_opt_dG1 = RT*np.log(true_opt_g1[:,0])
    return Km, v, estimate_opt_Km, true_opt_Km, estimate_opt_dG1, true_opt_dG1

def get_Km_v_inhibition(dG1,dGT,S,a,b,k10=1,k20=1):
    g1 = np.exp(dG1/RT)
    gT = np.exp(dGT/RT)
    sg1 = np.sqrt(g1)
    sgT = np.sqrt(gT)
    K = k20/k10/sgT
    Km = g1*(1+K)
    v = k10*K*sg1*S*ET/(a*Km + b*S)
    estimate_opt_Km = S
    true_opt_Km = S*b/a*np.ones(len(gT))
    true_opt_g1 = true_opt_Km/(1+K)
    true_opt_dG1 = RT*np.log(true_opt_g1[:,0])
    estimate_opt_dG1 = RT*(np.log(S) - np.log(1+K))[:,0]
    return Km, v, estimate_opt_Km, true_opt_Km, estimate_opt_dG1, true_opt_dG1

def get_Km_v_allosteric(dG1,dGT,S,n,k10=1,k20=1):
    g1 = np.exp(dG1/RT)
    gT = np.exp(dGT/RT)
    sg1 = np.sqrt(g1)
    sgT = np.sqrt(gT)
    K = k20/k10/sgT
    Km = g1*(1+K)
    v = k10*K*sg1*S**n*ET/(Km+S**n)
    opt_Km = S**(n)*np.ones(len(gT))
    opt_g1 = opt_Km/(1+K)
    opt_dG1 = RT*np.log(opt_g1[:,0])
    pred_opt_dG1 = RT*(np.log(S) - np.log(1+K))[:,0]
    return Km, v, opt_Km, opt_dG1, pred_opt_dG1