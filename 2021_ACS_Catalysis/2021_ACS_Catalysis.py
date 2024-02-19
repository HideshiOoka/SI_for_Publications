# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:20:53 2020
Last update: March 3rd, 2021
@author: Hideshi_Ooka
"""
# %%
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.lines as mlines

# UNIVERSAL CONSTANTS
FRT = 96485/8.314/300
ABC = ["A", "B","C","D","E"]
#
plt.rcParams["font.size"] = 20 
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["xtick.major.size"] =8.0
plt.rcParams["ytick.major.size"] = 8.0
plt.rcParams["xtick.major.width"] = 2.0
plt.rcParams["ytick.major.width"] = 2.0
plt.rc('legend', fontsize=14)
plt.rcParams['lines.markersize'] =3
AREA = np.pi * 0.55*0.55/4
MECHANISM ="VHT" # Choose between VHT or VT or VH
ALPHA_LOCK = 0.5# Set it to 0 to unlock alpha 

# %%
def get_log_j_VHT(par_vec):
    """
    The par_vec to be inputted:
    GH,logK2,logK3,a1, a2, a3 = par_vec
    GH: Binding Energy
    a1 to a3: Electron transfer coefficient/BEP coefficient of each elementary step
    """
    GH,logK2,logK3,a1, a2, a3 = par_vec
    K2 = 10**logK2
    K3 = 10**logK3
    if MECHANISM == "VT": # 
        K2 = 0
    if MECHANISM == "VH":
        K3 = 0
    if ALPHA_LOCK != 0:
        a1 = ALPHA_LOCK
        a2 = ALPHA_LOCK
        a3 = ALPHA_LOCK
    nFN = 1   
    k10 = 1
    k20 = k10 * K2
    k30 = k10 * K3
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
    j_VHT = 2*nFN*(k1**2*(k2+k3)-k1r**2*(k2r+k3r) +k1*k1r*(k2-k2r))/((k1+k1r)*(k1+k1r+k2+k2r+gamma)+2*(k1*k3+k1r*k3r))
    return np.log10(np.abs(j_VHT))

def get_contribution(par_vec):
    GH,logK2,logK3,a1, a2, a3 = par_vec
    K2 = 10**logK2
    K3 = 10**logK3 
    nFN = 1   
    k10 = 1
    k20 = k10 * K2
    k30 = k10 * K3
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
    netv1 = k1*(1-theta) - k1r*theta
    netv2 = k2*(theta) - k2r*(1-theta)
    netv3 = k3*(theta)**2 - k3r*(1-theta)**2
    
    return theta, np.log10(netv1),np.log10(netv2),np.log10(netv3)
    
    

def get_offset(log_j_VHT):
    offset = np.average(log_j_VHT - log_j_exp) # treat log_j_exp as a global parameter
    return offset

def get_offset_log_j_VHT(log_j_VHT):
    return log_j_VHT - get_offset(log_j_VHT)

def get_diff(par_vec):
    log_j_VHT = get_log_j_VHT(par_vec)
    diff = np.average((get_offset_log_j_VHT(log_j_VHT) - log_j_exp)**2)
    return diff

def get_log_diff(par_vec):
    log_j_VHT = get_log_j_VHT(par_vec)
    diff = np.average((get_offset_log_j_VHT(log_j_VHT) - log_j_exp)**2)
    log_diff = np.log10(diff)
    return log_diff

def get_log_j_VHT_keep_const_kzeros(par_vec):
    """
    The par_vec to be inputted:
    GH,logK2,logK3,a1, a2, a3 = par_vec
    GH: Binding Energy
    a1 to a3: Electron transfer coefficient/BEP coefficient of each elementary step
    """
    GH,logK2,logK3,a1, a2, a3 = par_vec
    K2 = 10**logK2
    K3 = 10**logK3
    k20 = np.sqrt(K2)
    k30 = np.sqrt(K3)
    if K2 < K3:
        k10 = 1/np.sqrt(K3)
    else:
        k10 = 1/np.sqrt(K2)
    nFN = 1   
    Eads = -GH
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
    j_VHT = 2*nFN*(k1**2*(k2+k3)-k1r**2*(k2r+k3r) +k1*k1r*(k2-k2r))/((k1+k1r)*(k1+k1r+k2+k2r+gamma)+2*(k1*k3+k1r*k3r))
    return np.log10(np.abs(j_VHT))
# %%
def crossover(parents):
    children = np.ones(shape = parents.shape)
    if ALPHA_LOCK != 0:
        for idx in range(parents.shape[1]-4):
            children[:,idx] = random.choices(parents[:,idx], weights = 1/10**parents[:,-1], k = parents.shape[0])
        children[:,-4:-1] = ALPHA_LOCK    
    else:
        for idx in range(parents.shape[1]-1):
            children[:,idx] = random.choices(parents[:,idx], weights = 1/10**parents[:,-1], k = parents.shape[0])
    return children

def evolve(parents):
    unique, count = np.unique(parents, axis=0, return_counts=True)
    duplicates = unique[count > 1]
    for duplicate in duplicates:
        repeated_idx = np.argwhere(np.all(parents == duplicate, axis = 1))
        evol_time = len(repeated_idx)
        evolved_parent = GD(duplicate, evol_time)
        parents[repeated_idx[0,0]] = evolved_parent
    return parents   

def GD(parent, evol_time):
    delta = 1E-5
    GD_cycles = 10 * evol_time
    old_par_vec = parent[0:-1] # actually has log_diff at the end
    old_diff = get_diff(old_par_vec)
    grad_vec = np.zeros(num_pars)
    
    if ALPHA_LOCK !=0:
        num_free_pars = num_pars -3
    else:
        num_free_pars = num_pars        
    for cycles in range(GD_cycles):
        for idx in range(num_free_pars):
            delta_vec = np.zeros(num_pars)
            delta_vec[idx] = delta
            # print(delta_vec[idx])
            grad_vec[idx] =(get_diff(old_par_vec+delta_vec) - old_diff)/delta

        new_par_vec = old_par_vec - 0.01 * grad_vec
        new_diff = get_diff(new_par_vec)
        if new_diff < old_diff:
            old_par_vec = new_par_vec
            old_diff = new_diff
        else:
            delta = delta/10
    old_log_diff = np.log10(old_diff)
    return np.append(old_par_vec, old_log_diff)

def make_Gaussian(filename):
    df = pd.read_csv(filename)
    M_Mean = df.mean()[:-4]
    M_STD = df.std()[:-4]
    
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_axes([0.2, 0.6,0.7,0.35]) 
    ax2 = fig.add_axes([0.2, 0.1,0.7,0.35]) 
    ax1.plot(GH_range, norm.pdf(GH_range, loc=M_Mean[0], scale = M_STD[0]), c = "r")
    ax2.plot(logK_range, norm.pdf(logK_range, loc=M_Mean[1], scale = M_STD[1]), c = "b", label = "$K_2$")
    ax2.plot(logK_range, norm.pdf(logK_range, loc=M_Mean[2], scale = M_STD[2]), c = "g", label = "$K_3$")
    ax1.set_xlabel("$\Delta G_\mathrm{H}$ [eV]")
    ax2.set_xlabel("log $K$ [-]")
    ax1.text(-0.2,1, "A", transform=ax1.transAxes,size=20, weight="bold")
    ax2.text(-0.2,1, "B", transform=ax2.transAxes,size=20, weight="bold")
    ax1.set_ylabel("Probability Density")
    ax2.set_ylabel("Probability Density")
    ax2.legend(frameon=False)
    return fig

def make_Tafel(filename):
    df = pd.read_csv(filename)
    par_vec = df.mean()[:-1]
    offset_log_j_VHT_2500 = get_offset_log_j_VHT(get_log_j_VHT(par_vec))
    
    
    fig = plt.figure(figsize = (8,4))
    ax1 = fig.add_axes([0.3, 0.2, 0.5, 0.7]) 
    ax1.scatter(E,log_j_exp, color = "k")
    ax1.plot(E,offset_log_j_VHT_2500, color = "r", linewidth = 3)
    ax1.set_xlabel("$E - iR$ [V vs. RHE]")
    ax1.set_ylabel("log $j$ [mA cm$^{-2}$]")
    ax1.set_xlim(-0.3,0.0)
    ax1.set_ylim(-0.2,3.5)
    ax1.set_xticks(np.arange(-0.3,0.01,0.1))
    ax1.set_yticks(np.arange(3,-0.1,-1))
    ax1.text(-0.27, 2.4, "$\Delta G_\mathrm{H}$ = " + str(par_vec[0])[:5], size=20, color = "r")
    return fig
    
#%%##########################################
############ IMPORT DATA ####################
#############################################
filename = "Pt_HER_1M.csv"
df = pd.read_csv(filename)
all_E = df.dropna()["Ecorr_3600"] # units: V vs. RHE after IR correction
all_j_exp = df.dropna()["i_3600"]/AREA # units: mA/cm2 geometric area of RDE 

j_exp = all_j_exp[all_E<-0.01].values # choose region for fitting
E = all_E[all_E < -0.01].values # choose region for fitting
all_E = all_E.values
all_j_exp = all_j_exp.values
log_j_exp = np.log10(np.abs(j_exp))
all_log_j_exp = np.log10(np.abs(all_j_exp))

# %%
pop_size = 1000
num_generations = 50
num_parents = int(pop_size * 0.5)
num_children = pop_size - num_parents 
num_pars = 6 # Eone,logKtwo,logKthree,a1,a2,a3
num_trials = 100
results_table = np.zeros((num_trials,num_pars+1))

for trial in range(num_trials):
    print(trial)
    pop = np.random.uniform(-0.95, 0.95, (pop_size,num_pars + 1)) #pars +  diff
    # change the scaling of each parameter
    pop[:,1] = pop[:,1] * 10 # -10 < logK2 <10
    pop[:,2] = pop[:,2] * 10 # -10 < logK2 <10
    pop[:,3:6] = 0.5 + pop[:,3:6]/2 # 0 < alpha <1
    if ALPHA_LOCK !=0: # Set ALPHA_LOCK to zero if alpha should be a free variable
        pop[:,3:6] = ALPHA_LOCK
    for p in range(pop_size):
        pop[p,-1] = get_log_diff(pop[p,0:-1])
    pop = pop[np.argsort(pop[:,-1])]
    pop_init = pop
    pop_dict= np.zeros((num_generations,pop_size,num_pars+1))
    for g in range(num_generations):
        pop_dict[g] = pop    
        parents = pop[0:num_parents]
        children = crossover(parents)
        evolved_parents = evolve(parents)
        pop = np.vstack((evolved_parents,children))
        for p in range(pop_size):
            pop[p,-1] = get_log_diff(pop[p,0:-1])
        pop = pop[np.argsort(pop[:,-1])]
        if pop[0,-1] == pop[-1,-1]:
            print("Optimization Converged at Generation : " + str(g))
            break
        plt.plot(pop[:,-1])
    results_table[trial] = pop[0]
# %%    
np.savetxt("results_table_" + MECHANISM +"_" + str(pop_size) +"_" + str(ALPHA_LOCK)+ ".csv", results_table, delimiter=',', header="GH,logK2,logK3,a1,a2,a3,log diff",comments='' )    
np.savetxt("pop_dict_" + MECHANISM +"_" + str(pop_size) +"_" + str(ALPHA_LOCK) + ".csv", pop_dict.reshape(pop_size*num_generations,num_pars+1), delimiter=',', )    


#%%# STATISTICS

pop_list = [10,30,50,100,300,500,1000]
M_Mean = np.zeros((len(pop_list), num_pars+2))
M_STD =  np.zeros((len(pop_list), num_pars+2))

for idx, pop in enumerate(pop_list):
    filename = ("results_table_VHT_pop_0.5.csv").replace('pop', str(pop))
    df = pd.read_csv(filename)
    M_Mean[idx,1:] = df.mean()
    M_STD[idx,1:] = df.std()
M_Mean[:,0] = pop_list
M_STD[:,0] = pop_list


np.savetxt("M_Mean_VHT_0.5.csv", M_Mean, delimiter=',', header="pop,GH, logK2, logK3, a1,a2,a3,log_diff",comments='' )      
np.savetxt("M_STD_VHT_0.5.csv", M_STD, delimiter=',', header="pop,GH, logK2, logK3, a1,a2,a3,log_diff",comments='' ) 
#%%##########################################
################ FIGURE 1 ###################
#############################################
fig = plt.figure(figsize=(8,4)) 
ax1 = fig.add_axes([0.2, 0.2, 0.3, 0.7])
ax1.scatter(all_E,all_j_exp,color = "r", s = 20)
ax1.set_xlim(-0.3, 0.2)
ax1.set_xticks(np.arange(-0.2,0.21,0.2))
ax1.set_xlabel("$E - iR$ [V vs. RHE]")  
ax1.set_ylim(-1800,100)
ax1.set_yticks(np.arange(0,-1501,-500))
ax1.set_ylabel("$j$  [mA cm$^{-2}$]") 

ax2 = fig.add_axes([0.65, 0.2, 0.3, 0.7]) 
ax2.scatter(all_E,all_log_j_exp, color = "r", s = 20)
ax2.plot(all_E,all_log_j_exp, color = "r")
ax2.set_xlim(-0.3, 0.2)
ax2.set_xticks(np.arange(-0.2,0.21,0.2))
ax2.set_xlabel("$E - iR$ [V vs. RHE]")  
ax2.set_ylim(-2,4)
ax2.set_yticks(np.arange(4,-2.9,-1))
ax2.set_ylabel("log $j$  [mA cm$^{-2}$]") 

axes = ax1,ax2
for idx, ax in enumerate(axes):
    ax.text(-0.35,0.95, ABC[idx], transform=ax.transAxes,size=20, weight="bold")    
plt.savefig('Fig1.png',dpi = 600)


#%%##########################################
################ FIGURE 2 ###################
#############################################
df=pd.read_csv("M_Mean_VHT_0.5.csv")
par_vec_opt = df.iloc[-1,1:-1].values # the first entry is the population
# log_j_theory_opt = get_log_j_VHT(par_vec_opt)
GH_opt = par_vec_opt[0]
par_vec_zero = par_vec_opt.copy()
par_vec_zero[0] = 0

log_j_VHT_opt = get_log_j_VHT(par_vec_opt)
offset_opt = get_offset(log_j_VHT_opt)
offset_log_j_VHT_opt = log_j_VHT_opt - offset_opt
offset_log_j_VHT_zero = get_log_j_VHT(par_vec_zero)- offset_opt

fig = plt.figure(figsize = (8,4))
ax1 = fig.add_axes([0.3, 0.2, 0.5, 0.7]) 
ax1.scatter(E,log_j_exp, color = "k")
ax1.plot(E,offset_log_j_VHT_opt, color = "r", linewidth = 3)
ax1.plot(E,offset_log_j_VHT_zero, color = "b", linewidth = 3)
ax1.set_xlabel("$E - iR$ [V vs. RHE]")
ax1.set_ylabel("log $j$ [mA cm$^{-2}$]")
ax1.set_xlim(-0.3,0.0)
ax1.set_ylim(-0.2,3.5)
ax1.set_xticks(np.arange(-0.3,0.01,0.1))
ax1.set_yticks(np.arange(3,-0.1,-1))
ax1.text(-0.27, 2.4, "$\Delta G_\mathrm{H}$ = " + str(GH_opt)[:5], size=20, color = "r")
ax1.text(-0.27, 1.2, "$\Delta G_\mathrm{H}$ = 0", size=20, color = "b")
plt.savefig("Fig2.png", dpi = 600)

#%%##########################################
################ FIGURE 3 ###################
#############################################

pop_dict = pd.read_csv('pop_dict_VHT_1000_0.5.csv', delimiter=',', header =None).values
pop_size = 1000
num_generations =50
num_pars = 6
pop_dict = pop_dict.reshape(num_generations,pop_size,num_pars+1) 
logK2_for_map = -10
matrix_size = 100
GH_array = np.linspace(-1,1,matrix_size+1)
logK3_array = np.linspace(-10,10,matrix_size+1)
log_diff = np.array([[get_log_diff((GH, logK2_for_map, logK3,0.5,0.5,0.5)) for GH in GH_array] for logK3 in logK3_array])
GH,logK3 = np.meshgrid(GH_array, logK3_array)

fig = plt.figure(figsize=(8,8))
x,y,w,h,sp = 0.15, 0.15, 0.3, 0.3,0.1
ax1 = fig.add_axes([x,y+h+sp,w,h]) 
ax2 = fig.add_axes([x+w+sp,y+h+sp,w,h]) 
ax3 = fig.add_axes([x,y,w,h]) 
ax4 = fig.add_axes([x+w+sp,y,w,h]) 
axes = ax1,ax2,ax3,ax4
for idx, ax in enumerate(axes):
    im = ax.contourf(GH,logK3,log_diff, 20, cmap = "rainbow")
    jdx = idx*10
    ax.scatter(pop_dict[jdx,:,0], pop_dict[jdx,:,2], color ="k", s=5)
    ax.text(0.05,0.85, ABC[idx], transform=ax.transAxes,size=20, weight="bold")
    ax.set_xlim(-1,1)
    ax.set_ylim(-10,10)
    ax.set_xticks(np.arange(-1,1.1,1))
    ax.set_yticks(np.arange(-10,10.1,5))

for jdx in range(pop_size):
    GH_grey = pop_dict[0,jdx,0]
    logK3_grey = pop_dict[0,jdx,2]
    if GH_grey < -0.6 and logK3_grey > 6:
        ax1.scatter(pop_dict[0,jdx,0], pop_dict[0,jdx,2], color ="grey", s=5)
fig.text(x+w+sp/2, 0.05, "$\Delta G_\mathrm{H}$ [eV]", ha = "center")  
fig.text(0.03, y+h+sp/2, "log $K_3$ [-]", va="center", rotation = "vertical")
cbar_ax = fig.add_axes([0.87, y, 0.02, 2*h+sp])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_title("log $R^2$")
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs,ha='right')
cbar.ax.yaxis.set_tick_params(pad=50)
plt.savefig('Fig3_again.png',dpi = 600)

#%%##########################################
################ FIGURE 4 ###################
#############################################
ls_list = ["solid",  "dashed", "dashdot", "dotted"]
M_Mean = pd.read_csv("M_Mean_VHT_0.5.csv")
M_Mean["pop"] = M_Mean["pop"].astype(int)
M_STD = pd.read_csv("M_STD_VHT_0.5.csv")
GH_range = np.linspace(-0.4, 0.4, 401)
logK_range = np.linspace(-15, 5, 401)

fig = plt.figure(figsize = (8,8))
ax1 = fig.add_axes([0.2, 0.6,0.7,0.35]) 
ax2 = fig.add_axes([0.2, 0.1,0.7,0.35]) 
scale = 0.5

for idx in range(4):
    if idx >0:
        scale = 1
    ax1.plot(GH_range, scale *norm.pdf(GH_range, loc=M_Mean.iloc[-idx-1,1], scale = M_STD.iloc[-idx-1,1]), label = str(M_Mean.iloc[-idx-1,0]), c = "r", ls = ls_list[idx])
    ax2.plot(logK_range, norm.pdf(logK_range, loc=M_Mean.iloc[-idx-1,2], scale = M_STD.iloc[-idx-1,2]), label = str(M_Mean.iloc[-idx-1,0]), c = "b", ls = ls_list[idx])
    ax2.plot(logK_range, norm.pdf(logK_range, loc=M_Mean.iloc[-idx-1,3], scale = M_STD.iloc[-idx-1,3]), c= "g", ls = ls_list[idx])
ax1.set_xlabel("$\Delta G_\mathrm{H}$ [eV]")
ax2.set_xlabel("log $K$ [-]")
ax1.text(-0.2,1, "A", transform=ax1.transAxes,size=20, weight="bold")
ax2.text(-0.2,1, "B", transform=ax2.transAxes,size=20, weight="bold")
ax1.legend(loc = "upper left", frameon  = False)
ax2.text(-8.5,1.2, "$K_2$",size=20, weight='bold', color = "b")
ax2.text(-5,0.9, "$K_3$",size=20, weight='bold', color = "g")
ax1.set_ylabel("Probability Density")
ax2.set_ylabel("Probability Density")
plt.savefig("Fig4.png", dpi = 600)

#%%##########################################
################ FIGURE 5 ###################
#############################################
fig = plt.figure(figsize = (8,4))
ax1 = fig.add_axes([0.2, 0.25,0.3,0.65]) 
ax2 = fig.add_axes([0.65, 0.25,0.3,0.65]) 
GH = 0.1
w = 1
s = 0.5
off = 0.05
GH_list = [0,0.1,-0.1]
c_list = ["k", "r","b"]
logK_list = [5,-5] # [0,5,-5]
ls_list = ["solid","dashed"]
marker_list = ["", "o"] 
E = np.linspace(-0.001, -1, 30)
for idx, GH in enumerate(GH_list):
    ax1.plot([w,w+s], [0,GH], color = c_list[idx])
    ax1.plot([w+s+off,2*w+s-off], [GH,GH], color = c_list[idx], linewidth=5)
    ax1.plot([2*w+s,2*w+2*s], [GH,0], color = c_list[idx])
    for jdx, logK in enumerate(logK_list):
        ax2.plot(E, get_log_j_VHT_keep_const_kzeros((GH,-30, logK, 0.5,0.5,0.5)), color = c_list[idx], linestyle = ls_list[jdx])
ax1.plot([0,w-off], [0,0], "k", linewidth=5)
ax1.plot([2*w+2*s+off,3*w+2*s], [0,0], "k", linewidth=5)
ax1.set_xlabel("Reaction Coordinates")
ax1.set_ylabel("$\Delta G_\mathrm{H}$ [eV]")
ax1.set_xticks([])
ax1.set_ylim([-0.15, 0.15])
ax1.text(0.5*w, -0.01, "H$^+$ + e$^-$", va="top", ha = "center", size=10)
ax1.text(1.5*w+s, -0.01, "H$_{\mathrm{ads}}$", va="top", ha = "center", size=10)
ax1.text(2.5*w+2*s, -0.01, r"$\frac{1}{2}$ H$_{\mathrm{2}}$", va="top", ha = "center", size=10)

ax2.set_xlabel("$E$ / V vs. RHE")
ax2.set_ylabel("log | $j$ |")
ax2.set_xlim(-1,0)
ax2.set_ylim(-5,4)
ax2.set_xticks([-1,-0.5,0])
ax2.set_yticks(np.linspace(-4,4,5))

ax1.text(-0.25,1, "A", transform=ax1.transAxes,size=20, weight="bold")
ax2.text(-0.25,1, "B", transform=ax2.transAxes,size=20, weight="bold")
solid = mlines.Line2D([], [], color="k", linestyle="solid", label="log $K_3$ = 5")
dashed= mlines.Line2D([], [], color="k", linestyle="dashed", label="log $K_3$ = -5")
ax2.legend(handles=[solid,dashed],frameon=False, facecolor = "none",fontsize = 10)
plt.savefig("Fig5.png", dpi = 600)



#%%##########################################
########## SI FIGURES START HERE ############
#############################################

#############################################
######## FIGURES ON MASS TRANSPORT ##########
#############################################

### pH dependence of jlim
df = pd.read_csv("jlim_pH_dependence.csv")
c_list = ["k", "r","b", "g"]
pH_list = [0.34, 1.72, 2.77, 3.8]
# label_list = ["1 M (pH 0.34)", "100 mM (pH 1.72)", "10 mM (pH 2.77)", "1 mM (pH 3.8)"]
label_list = ["1 M", "100 mM", "10 mM", "1 mM"]

# H_arr = np.array([1.72, 2.77, 3.8])
pH_arr = np.array([0.34, 1.72, 2.77, 3.8])
H_arr = 10**(-pH_arr) * 1000 # mM unit
j_lim_arr = np.array([])

fig = plt.figure(figsize = (8,4))
ax1 = fig.add_axes([0.12, 0.2,0.35,0.7]) 
ax2 = fig.add_axes([0.63, 0.2,0.35,0.7]) 
for idx in range(4):
    E_RHE = df.iloc[:,2*idx] + 0.2 + 0.059 * pH_list[idx] # this is IR corrected already, convert Ag/AgCl to RHE
    I = df.iloc[:,2*idx+1] # A units

    j = I * 1000/AREA # mA/cm2 units,
    j_lim = -np.mean(j[ (E_RHE<-0.24) & (E_RHE > -0.25) ])
    j_lim_arr = np.append(j_lim_arr, j_lim)
    ax1.scatter(E_RHE,plt.np.log10(np.abs(j)), c= c_list[idx], label = label_list[idx], s = 3)
    # ax.plot(E_RHE,j, c_list[idx])# , label = label_list[idx])
    # print(np.log10(np.abs(j))[E_RHE <-0.2])

ax1.set_xlim(-0.25, 0.2)
ax1.set_xlabel("$E - iR$ [V vs.RHE]")
ax1.set_ylabel("log $j$  [mA cm$^{-2}$]")
ax1.legend(loc = "upper right", frameon  =False, markerscale = 2, fontsize = 12)
ax2.scatter(H_arr, j_lim_arr, c = "k", s = 20)
ax2.set_ylim(-10, 150)
ax2.set_xlim(-10,50)
ax2.set_xlabel("[H$^+]$ [mM]")
ax2.set_ylabel("$|j_{\mathrm{lim}}|$ [mA/cm$^2$]" )

ATA = np.dot(H_arr[1:], H_arr[1:])
ATb = np.dot(H_arr[1:], j_lim_arr[1:])
ax2.plot(H_arr, ATb/ATA * H_arr, "r")

ax2.plot([0,H_arr[0]], [0,j_lim_arr[0]], "b--")
# H_arr[0] = 457
# j_lim_arr[0] = 1479

ax1.text(-0.3,1, "A", transform=ax1.transAxes,size=20, weight="bold")
ax2.text(-0.35,1, "B", transform=ax2.transAxes,size=20, weight="bold")
plt.savefig("jlim_vs_pH.png", dpi  = 600)

#%% RPM dependence of 1 M H2SO4 + 2 M Na2SO4
df = pd.read_csv("Pt_HER_1M.csv")
clist = ["black","royalblue", "blue", "green", "grey", "orange", "red", "darkred"]
rpm_list = [400,493,625,816,1111,1600,2500,3600]
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_axes([0.15, 0.2, 0.3, 0.7]) 
ax2 = fig.add_axes([0.6, 0.2, 0.3, 0.7]) 

for idx in range(8):
    E = df.iloc[:,2*idx]
    j = df.iloc[:,2*idx+1]/AREA
    log_j = np.log10(np.abs(j))
    ax1.scatter(E,j, color = clist[idx], label = str(rpm_list[idx]))
    ax2.plot(E,log_j, clist[idx], label = str(rpm_list[idx]))
    
    ax1.set_xlim(-0.3,0.1)
    ax1.set_xticks(np.arange(-0.2,0.01,0.2))
    ax1.set_xlabel("$\it{E-iR}$ [V vs. RHE]")
    ax1.set_ylabel("$\it{j}$ [mA cm$^{-2}$]")
    ax1.text(0.85, 0.85, "A", transform=ax1.transAxes,  fontsize=20, fontweight='bold', va='top')
    
    ax2.set_xlim(-0.3,0.1)
    ax2.set_xticks(np.arange(-0.2,0.01,0.2))
    ax2.set_xlabel("$\it{E-iR}$ [V vs. RHE]")
    ax2.set_ylabel("log $\it{j}$ [mA cm$^{-2}$]")
    ax2.text(0.85, 0.85, "B", transform=ax2.transAxes,  fontsize=20, fontweight='bold', va='top')
    ax1.legend(loc="lower right", frameon = False, markerscale = 2, fontsize = 10)
plt.savefig('CV_rpm.png',dpi = 600)

#%%#############################
df = pd.read_csv("KL_manual.csv")
rpm_j = df.drop(columns="Ecorr_400").values

fig, (ax1,ax2) = plt.subplots(1, 2)
plt.tight_layout(w_pad=2)
fig.set_size_inches(8,4)
clist = ["r", "b", "g","k"]

E_list = [-0.25, -0.2, -0.15, -0.1]

for i in range(4):
    ax1.scatter(np.sqrt(rpm_list), rpm_j[i]/AREA, c=clist[i], label = str(E_list[i]), s = 20)
    ax2.scatter(1/np.sqrt(rpm_list), 1/rpm_j[i]*AREA,c=clist[i], s = 20)
    ax1.text(0.85, 0.95, "A", transform=ax1.transAxes,  fontsize=16, fontweight='bold', va='top')
    ax2.text(0.85, 0.95, "B", transform=ax2.transAxes,  fontsize=16, fontweight='bold', va='top')
    
    ax1.set_xlim(0,65)
    ax1.set_xticks(np.arange(0,61,20))
    ax1.set_ylim(-1600,100)
    ax1.set_yticks(np.arange(0,-1501,-500))
    ax1.set_xlabel("$\omega^{0.5}$ [rpm]")
    ax1.set_ylabel("$\it{j}$ [mA cm$^{-2}$]")
    
    
    ax2.set_xlim(0,0.06)
    ax2.set_xticks(np.arange(0,0.061,0.02))
    ax2.set_ylim(-0.065,0.01)
    ax2.set_yticks(np.arange(-0.06,0.01,0.02))
    ax2.set_xlabel("$\omega^{-0.5}$ [rpm]")
    ax2.set_ylabel("$\it{j}^{-1}$ [mA cm$^{-2}$]")
    ax1.legend(fontsize=12,bbox_to_anchor=(0.05, 0.05), loc='lower left', frameon = False)
plt.savefig('Levich_KL.png',dpi = 600, bbox_inches='tight')

#%%##########################################
######## EVALUATE MASS TRANSPORT (INDIRECT)
#############################################

### FITTING OF 2500 rpm data
filename = "Pt_HER_1M.csv"
df = pd.read_csv(filename)
all_E = df.dropna()["Ecorr_2500"] # units: V vs. RHE after IR correction
all_j_exp = df.dropna()["i_2500"]/AREA # units: mA/cm2 geometric area of RDE 

j_exp = all_j_exp[all_E<-0.01].values # choose region for fitting
E = all_E[all_E < -0.01].values # choose region for fitting
log_j_exp = np.log10(np.abs(j_exp))

"""""
# do the fitting 
np.savetxt("results_table_2500_" + MECHANISM +"_" + str(pop_size) +"_" + str(ALPHA_LOCK)+ ".csv", results_table, delimiter=',', header="GH,logK2,logK3,a1, a2, a3, log diff",comments='' )    
np.savetxt("pop_dict_2500_" + MECHANISM +"_" + str(pop_size) +"_" + str(ALPHA_LOCK) + ".csv", pop_dict.reshape(pop_size*num_generations,num_pars+1), delimiter=',', )    
GH       0.091897
logK2   -9.221117
logK3   -4.819352
a1       0.500000
a2      0.500000
a3      0.500000
"""

#### Show the figure
fig = make_Tafel("results_table_2500_VHT_1000_0.5.csv")
plt.savefig("Tafel_Plot_2500.png", dpi= 600)

###Gaussian
GH_range = np.linspace(0, 0.2, 401)
logK_range = np.linspace(-15, 0, 401)
fig = make_Gaussian("results_table_2500_VHT_1000_0.5.csv")
# fig.axes[0].text(-8.5,1.3, "$K_2$",size=20, weight='bold', color = "b")
# fig.text(-5,1.1, "$K_3$",size=20, weight='bold', color = "g")
plt.savefig("Gaussian_2500.png", dpi = 600)

# %% FITTING OF NON-BENDING REGION
filename = "Pt_HER_1M.csv"
df = pd.read_csv(filename)
all_E = df.dropna()["Ecorr_3600"] # units: V vs. RHE after IR correction
all_j_exp = df.dropna()["i_3600"]/AREA # units: mA/cm2 geometric area of RDE 

wide_j_exp = all_j_exp[all_E<-0.01].values # choose region for fitting
wide_E = all_E[all_E < -0.01].values # choose region for fitting
wide_log_j_exp = np.log10(np.abs(wide_j_exp))
restricted_j_exp = all_j_exp[(all_E<-0.01) & (all_E > -0.05)].values
restricted_E = all_E[(all_E<-0.01) & (all_E > -0.05)].values
log_restricted_j_exp = np.log10(np.abs(restricted_j_exp))
### afterwards, use this data to fit as normal
### Results will be plotted below
df = pd.read_csv("results_table_restrictedE_VHT_1000_0.5.csv")
M_Mean = df.mean()
M_STD = df.std()
"""
GH           0.646653
logK2       -9.437947
logK3       -0.228197
a1           0.500000
a2          0.500000
a3          0.500000
"""
par_vec_restricted = M_Mean[:-1]
E = restricted_E
log_j_VHT_restricted = get_log_j_VHT(par_vec_restricted)
offset = np.average(log_j_VHT_restricted - log_restricted_j_exp)
E = wide_E
offset_log_j_VHT_restricted = get_log_j_VHT(par_vec_restricted) - offset


fig = plt.figure(figsize = (8,4))
ax1 = fig.add_axes([0.3, 0.2, 0.5, 0.7]) 
ax1.scatter(wide_E,log_wide_j_exp, color = "grey")
ax1.scatter(restricted_E,log_restricted_j_exp, color = "k")
ax1.plot(wide_E,offset_log_j_VHT_restricted, color = "r", linewidth = 3)
ax1.set_xlabel("$E - iR$ [V vs. RHE]")
ax1.set_ylabel("log $j$ [mA cm$^{-2}$]")
ax1.set_xlim(-0.3,0.0)
ax1.set_ylim(-0.2,8)
ax1.set_xticks(np.arange(0,-0.31,-0.1))
ax1.set_yticks(np.arange(0,8.1,2))
plt.savefig("Tafel_Plot_Restricted.png", dpi= 600)

###### GAUSSIAN

GH_range = np.linspace(0, 1, 401)
logK_range = np.linspace(-15, 5, 401)
make_Gaussian("results_table_restrictedE_VHT_1000_0.5.csv")
# ax2.text(-8.5,1.2, "$K_2$",size=20, weight='bold', color = "b")
# ax2.text(0,0.9, "$K_3$",size=20, weight='bold', color = "g")
plt.savefig("Gaussian_Restricted.png", dpi = 600)


#%%##########################################
######## EVALUATE ALPHA #####################
#############################################


### FREE ALPHA
fig = make_Tafel("results_table_VHT_5000_0.csv")
plt.savefig("Tafel_Plot_Free_Alpha.png", dpi= 600)

GH_range = np.linspace(-0.5, 1, 401)
logK_range = np.linspace(-15, 10, 401)
fig = make_Gaussian("results_table_VHT_5000_0.csv")
plt.savefig("Gaussian_Free_Alpha.png", dpi = 600)

### ALPHA = 0.6
fig = make_Tafel("results_table_VHT_1000_0.6.csv")
plt.savefig("Tafel_Plot_Alpha_0.6.png", dpi= 600)

GH_range = np.linspace(0, 0.2, 401)
logK_range = np.linspace(-15, 0, 401)
fig = make_Gaussian("results_table_VHT_1000_0.6.csv")
plt.savefig("Gaussian_Alpha_0.6.png", dpi = 600)

### ALPHA = 0.4
fig = make_Tafel("results_table_VHT_1000_0.4.csv")
plt.savefig("Tafel_Plot_Alpha_0.4.png", dpi= 600)

GH_range = np.linspace(0, 0.2, 401)
logK_range = np.linspace(-15, 0, 401)
fig = make_Gaussian("results_table_VHT_1000_0.4.csv")
plt.savefig("Gaussian_Alpha_0.4.png", dpi = 600)


#%%##########################################
###### OTHER MECHANISTIC FIGURES ############
#############################################


### FITTING OF VH MECHANISM
fig = make_Tafel("results_table_VH_1000_0.5.csv")
plt.savefig("Tafel_Plot_VH.png", dpi= 600)

GH_range = np.linspace(-0.5, 1, 401)
logK_range = np.linspace(-15, 10, 401)
fig = make_Gaussian("results_table_VH_1000_0.5.csv")
plt.savefig("Gaussian_VH.png", dpi = 600)



#%% COVERAGE AND CONTRIBUTION
c = get_contribution(par_vec_opt)
label_list = ["theta", "netv1", "netv2", "netv3"]
c_list = ["b", "g"]

fig = plt.figure(figsize = (8,8))
ax1 = fig.add_axes([0.2, 0.55, 0.6, 0.4]) 
ax2 = fig.add_axes([0.2, 0.15, 0.6, 0.4]) 
ax1r=ax1.twinx()
ax2r=ax2.twinx()
ax1r.set_ylabel(r"$ \theta_{H} $", color = "r")

ax1.scatter(E,log_j_exp, color = "k")
ax1r.plot(E,c[0],"ro")
ax2r.scatter(E,c[2]/c[1]*100, c="r")

offset = get_offset(c[1])



for idx in range(2,4):
    ax2.plot(E,c[idx]-offset, c = c_list[idx-2], label = label_list[idx])



ax1.set_ylabel("$j$  [mA cm$^{-2}$]") 

ax1r.set_yticks(np.linspace(0,1,3))
ax1.set_xticklabels([])
for ax in (ax1r, ax2r):
    ax.set_xlim(-0.2,0)
    ax.spines["right"].set_color('red')
    ax.xaxis.label.set_color('red')
    ax.tick_params(axis='y', colors='red')
    ax.tick_params(axis='y', pad=15)
ax2.tick_params(axis='x', pad=8)    
ax2.set_xlabel("$E - iR$ [V vs. RHE]")  
ax2.set_ylabel("$j_\mathrm{partial}$ [mA cm$^{-2}$]")
ax2r.set_ylabel("Contribution of\n Heyrovsky Step [%]",color="r")
ax2r.set_ylim(0,0.049)
ax1.text(-0.2,1, "A", transform=ax1.transAxes,size=20, weight="bold")
ax2.text(-0.2,1, "B", transform=ax2.transAxes,size=20, weight="bold")
ax2.text(-0.19,-0.5, "Volmer-Heyrovsky",size=20, color = "b")
ax2.text(-0.19,2, "Volmer-Tafel",size=20, color = "g")
plt.savefig("Contribution.png", dpi = 600)




#%%############################
##OTHER SUPPLEMENTARY FIGURES## 
###############################

df = pd.read_csv("EIS_E_Dependence.csv").dropna()
cols = df.shape[1]
potential_list = list(df.columns.values)[0:cols:2]
fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0.3, 0.15, 0.6, 0.8]) 
clist = ["blue", "deepskyblue", "green", "grey", "orange", "red"]
for i in range(int(cols/2)):
    ax.scatter(df.iloc[:,2*i],df.iloc[:,2*i + 1], c=clist[i], label='_nolegend_')    
    ax.plot(df.iloc[:,2*i],df.iloc[:,2*i + 1], c=clist[i], label = potential_list[i])    
    ax.set_xlim(0,6)
    ax.set_xticks(np.arange(0,7,2))
    ax.set_ylim(1,-6)
    ax.set_yticks(np.arange(0,-7,-2))
    ax.set_xlabel("$Z_\mathrm{re}$ [$\Omega$]")
    ax.set_ylabel("$Z_\mathrm{im}$ [$\Omega$]")
    ax.legend(fontsize=12)
plt.tight_layout
plt.savefig('EIS_E_Dependence',dpi = 600)    


#%% POPULATION DEPENDENCE OF FITTING ACCURACY
M_Mean = pd.read_csv("M_Mean_VHT_0.5.csv", index_col = 0)
M_STD = pd.read_csv("M_STD_VHT_0.5.csv", index_col = 0)
pop_array = M_Mean.index.values
fmt_list = ["ro", "bo", "go", "ko","ko","ko","ko"]

fig = plt.figure(figsize = (8,8))
x,y,w,h,vsp = 0.2, 0.1, 0.7, 0.29,0
ax1 = fig.add_axes([x,y+2*h+2*vsp,w,h]) 
ax2 = fig.add_axes([x,y+h+vsp,w,h]) 
ax3 = fig.add_axes([x,y,w,h]) 
for idx in range(num_pars+1):
    if idx ==0:
        ax = ax1
    elif idx < 3:
        ax = ax2
    elif idx < 6:
        continue        
    elif idx == 6:
        ax = ax3        
    for pop, mean, std in zip(pop_array, M_Mean.iloc[:,idx], M_STD.iloc[:,idx]):
        ax.errorbar(pop, mean, std, fmt = fmt_list[idx], capsize = 8)
    ax1.set_ylabel("$\Delta G_\mathrm{H}$ [eV]")  
    ax1.set_ylim(-0.8,1)
    ax1.axhline(y = 0.094, c = "r", ls = "--", lw = 1)
    ax2.set_ylabel("$K$ [-]")    
    ax2.set_ylim(-12,12)
    ax3.set_ylabel("log $R^2$ [-]")
    ax3.set_ylim(-2.2,-0.8)
    ax3.set_yticks([-2,-1])
    ax3.set_xlabel("Population")
    ax1.text(0.9,0.8, "A", transform=ax1.transAxes,size=20, weight="bold")       
    ax2.text(0.9,0.8, "B", transform=ax2.transAxes,size=20, weight="bold")       
    ax3.text(0.9,0.8, "C", transform=ax3.transAxes,size=20, weight="bold")   
    ax2.text(200, -8, "$K_2$", c = "b")
    ax2.text(200, -4, "$K_3$", c = "g")
    
    for ax in (ax1,ax2,ax3):
        ax.set_xscale("log")
        ax.yaxis.set_label_coords(-0.13, 0.5)
plt.savefig("Mean_STD_vs_Population.png", dpi = 600)


#%% ML CYCLE DEPENDENCE

pop_size = 1000
num_generations = 50
num_pars =6
df = pd.read_csv("pop_dict_VHT_1000_0.5.csv", header = None)
pop_dict = df.values.reshape(num_generations,pop_size,num_pars+1)
import itertools



# find final generation
for idx in range(num_generations):
    # print(pop_dict[idx])
    if (pop_dict[idx] == np.zeros((pop_size,num_pars+1))).all():
        max_gen = idx
        break


GH_arr = pop_dict[:max_gen,:,0].flatten()
logK2_arr = pop_dict[:max_gen,:,1].flatten()
logK3_arr = pop_dict[:max_gen,:,2].flatten()
log_diff_arr = pop_dict[:max_gen,:,-1].flatten()
gen_arr = list(itertools.chain.from_iterable(itertools.repeat(x, pop_size) for x in range(max_gen)))

fig = plt.figure(figsize = (8,8))
x,y,w,h,vsp = 0.2, 0.1, 0.7, 0.29,0
ax1 = fig.add_axes([x,y+2*h+2*vsp,w,h]) 
ax2 = fig.add_axes([x,y+h+vsp,w,h]) 
ax3 = fig.add_axes([x,y,w,h]) 
ax1.scatter(gen_arr, log_diff_arr, c="k", alpha = 0.5, s = 1)
ax2.scatter(gen_arr, GH_arr, c="r", alpha = 0.5, s = 1)
ax3.scatter(gen_arr, logK2_arr, c="b", alpha = 0.5, s = 1)
ax3.scatter(gen_arr, logK3_arr, c="g", alpha = 0.5, s = 1)
ax1.set_xlabel([])
ax2.set_xlabel([])
ax1.set_ylim(-2.2,0.5)
ax2.set_ylim(-1.3,1.3)
ax3.set_ylim(-11,11)



ax1.set_ylabel("log $R^2$ [-]",labelpad=10)
ax2.set_ylabel("$\Delta G_\mathrm{H}$ [eV]",labelpad=10)
ax3.set_ylabel("log $K$ [-]",labelpad=0)
ax3.set_xlabel("Machine Learning Cycles [-]")
ax3.text(30,logK2_arr[-1]+1, "$K_2$",size=20, weight='bold', color = "b")
ax3.text(30,logK3_arr[-1]+1, "$K_3$",size=20, weight='bold', color = "g")
ax1.text(-0.17,0.9, "A", transform=ax1.transAxes,size=20, weight="bold")
ax2.text(-0.17,0.9, "B", transform=ax2.transAxes,size=20, weight="bold")
ax3.text(-0.17,0.9, "C", transform=ax3.transAxes,size=20, weight="bold")

plt.savefig('ML_Efficiency.png',dpi = 600)



    
