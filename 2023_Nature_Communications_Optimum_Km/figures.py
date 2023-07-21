#%%
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:28:47 2022
@author: Hideshi_Ooka
"""
import matplotlib.pyplot as plt
import numpy as np
from enzyme_kinetics import get_Km_v, get_Km_v_inhibition, get_Km_v_reverse, get_Km_v_allosteric, get_dG1_from_Km
from bioinformatics import classify_Km_S_data, fit_Gaussian
#%%
#######################################
######### MATPLOTLIB SETTINGS #########
#######################################
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
plt.rcParams["xtick.minor.size"] =8.0
plt.rcParams["ytick.minor.size"] = 8.0
plt.rcParams["xtick.minor.width"] = 2.0
plt.rcParams["ytick.minor.width"] = 2.0
plt.rc('legend', fontsize=14)
plt.rcParams['lines.markersize'] =3
k = "#141418"
b = "#083E73"
r = "#B10026"
g = "#228B38"
lb = "#3C93C2"
grey = "#4B5450"

#######################################
### CONTOUR PLOT PARAMETERS ###########
#######################################
MIN,MAX  = -9.5,2 # colors for the contour plots
arr_size = 201 # number of datapoints in meshgrid
scale = 20 # scale of delta G1
dG1_range = np.linspace(-scale, scale, arr_size) # kJ/mol
dGT_range = np.linspace(-scale*2, scale*2, arr_size)
dG1_mesh, dGT_mesh = np.meshgrid(dG1_range,dGT_range)

##################################
### Fig.1,2 was made using PPT #####
##################################
#%%###############################
### Fig.3 ########################
##################################

def get_Ea_curve(x1,y1,x2,y2):
    # This will draw a 3rd order polynomial between (x1,y1) and (x2,y2) which is smooth at both ends
    # The equation is given in y
    a = 6*(y2-y1)/(x1-x2)**3
    C = y1 + a/6*x1**2*(x1-3*x2)
    # C2 = y2 + a/6*x2**2*(x2-3*x1) # this should be equal to C    
    x = np.linspace(x1,x2)
    y = a*(x**3/3 -(x1+x2)/2*x**2 + x1*x2*x)+C
    return x,y

def draw_landscape(dG1, dGT, ax, c = "k", Ea0=25, w = 0.4):
    Ea1 = Ea0 + 0.5*dG1
    Ea2 = Ea0 + 0.5*(dGT-dG1)
    xs = np.arange(0,5)
    ys = [0, Ea1, dG1, Ea2+dG1, dGT]
    N = len(xs)
    landscape_x = np.array([])
    landscape_y = np.array([])
    for i in range(N-1):
        x,y = get_Ea_curve(xs[i],ys[i],xs[i+1],ys[i+1])
        landscape_x = np.append(landscape_x,x)
        landscape_y = np.append(landscape_y,y)
        if i == 2: # it is ES
            ax.plot([xs[i]-w, xs[i]+w], [ys[i],ys[i]], c, lw = 3)
    ax.plot(landscape_x,landscape_y, c, alpha = 0.5, lw = 2)
    ax.plot([xs[0]-w, xs[0]+w], [ys[0],ys[0]], k, lw = 3)
    ax.plot([xs[-1]-w, xs[-1]+w], [ys[-1],ys[-1]], k, lw = 3)


def draw_Fig3(dG1_list, dGT, S, h = -45, ylim = 1):
    c_list = [k,b,r,g]
    ha_list = ["center","left","center"]
    fig = plt.figure(figsize = (8,4))
    ax1 = fig.add_axes([0.13, 0.2,0.35,0.7]) 
    ax2 = fig.add_axes([0.63, 0.2,0.35,0.7]) 
    for i, dG1 in enumerate(dG1_list):
        draw_landscape(dG1_list[i], dGT, ax1, c_list[i])    
        Km, v, estimate_opt_Km, true_opt_Km, estimate_opt_dG1, true_opt_dG1 = get_Km_v(dG1,dGT,S)
        if dGT >=0: # This section is for Fig. S3, which has a too small y-axis because the entire reaction is unfavorable
            v = v*100
            if i == 0:
                ax2.text(0.98,0.9,r"$\times$ 100", transform=ax2.transAxes, fontsize = 14, ha = "right")
                ylim = ylim*100
        ax2.plot(S,v,c_list[i], marker= "o", markersize = 3)
        ax2.axvline(Km,c = c_list[i], ls = "--", lw = 2)
        ax2.annotate('', xy=[Km,ylim*0.8], xytext=[Km,ylim*0.9],
                    arrowprops=dict(width=1, headwidth=8, 
                                    color = c_list[i]))
        ax2.text(Km,ylim*1.05, "$K_\mathrm{m}$", c = c_list[i], ha = ha_list[i])
    ax1.text(0,h, "E+S", ha = "center", va = "top")
    ax1.text(2,h, "ES", ha = "center", va = "top")
    ax1.text(4,h, "E+P", ha = "center", va = "top")
    ax1.set_ylabel("$\Delta G$ [kJ/mol]")
    ax1.set_xticks([])
    ax1.set_xlim(-0.5,4.5)
    ax2.set_xlim(-0.5,S.max())
    ax2.set_ylim(0,ylim)
    ax2.set_xlabel("[S] [$\mu$M]")
    ax2.set_ylabel("$v$ [$\mu$M/s]")
    if dGT == -40:
        ax1.text(-0.35,0.96,"A", transform=ax1.transAxes, fontsize = 20, weight = "bold")
        ax2.text(-0.32,0.96,"B", transform=ax2.transAxes, fontsize = 20, weight = "bold")
        ax2.set_xticks(np.arange(S.max()+1)[::2])
        plt.savefig("dG_and_MM.pdf")
    else:
        ax1.text(-0.35,1.05,"a", transform=ax1.transAxes, fontsize = 20, weight = "bold")
        ax2.text(-0.32,1.05,"b", transform=ax2.transAxes, fontsize = 20, weight = "bold")
        plt.savefig(f"dG_and_MM_{dGT}.pdf")
    plt.show()
dGT = -2*scale # kJ/mol
Smax=8
S_lin = np.linspace(0,Smax,101)
draw_Fig3([-25,-20,-15], dGT, S_lin)
#%%###############################
### Fig.4 ########################
##################################
def make_contours(X, Y, Z, 
                  X_label = "$\Delta G_1$ [kJ/mol]", 
                  Y_label = "$\Delta G_\mathrm{T}$ [kJ/mol]"):
    """
    This function returns the 4 panel contour plot, but annotations differ between figures so they are outside the function.
    """
    contour_levels = np.linspace(MIN,MAX,51)
    X_scale = X.max()
    Y_scale = Y.max()
    fig = plt.figure(figsize=(8,6.4))
    x,y,w,h,hsp,vsp = 0.15, 0.15, 0.25, 0.3125,0.1,0.125
    ax1 = fig.add_axes([x,y+h+vsp,w,h]) 
    ax2 = fig.add_axes([x+w+hsp,y+h+vsp,w,h]) 
    ax3 = fig.add_axes([x,y,w,h]) 
    ax4 = fig.add_axes([x+w+hsp,y,w,h]) 
    axes = ax1,ax2,ax3,ax4
    for i in range(4):
        ax = axes[i]
        im = ax.contourf(X,Y, Z[i], 50, cmap = "jet", levels = contour_levels)
        ax.contour(X,Y, Z[i], levels = contour_levels, linewidths = 0.3, colors=[(0,0,0,0.5)])
        ax.set_xticks([-X_scale,0,X_scale])
        ax.set_yticks([-Y_scale,0,Y_scale])
        ax.text(0.05,0.8,"abcd"[i], transform=ax.transAxes, fontsize = 20, weight = "bold", color = k)
        if i >= 2:
            ax.set_xlabel(X_label)  
        if i%2 ==0:
            ax.set_ylabel(Y_label)   
        ax.set_xlim(-X_scale, X_scale)
        ax.set_ylim(-Y_scale, Y_scale)
    cbar_ax = fig.add_axes([x+hsp*2+2*w, y, 0.02, 2*h+vsp])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks = np.arange(int(MIN)+1,int(MAX)+1,2))
    cbar_ax.set_title("log $v$ [$\mu$M/s]", fontsize = 20, pad = 20)
    ticklabs = cbar_ax.get_yticklabels()
    cbar_ax.yaxis.set_tick_params(pad=10)
    return fig,axes,cbar_ax

S_list = [0.1,1,10,100] # microM units
log10v_arr = np.zeros((4,arr_size,arr_size)) # 4 panels
estimate_opt_dG1_arr = np.zeros((4,arr_size))
for i,S in enumerate(S_list):
    Km, v, estimate_opt_Km, true_opt_Km, estimate_opt_dG1, true_opt_dG1 = get_Km_v(dG1_mesh,dGT_mesh,S)
    log10v_arr[i] = np.log10(v)
    estimate_opt_dG1_arr[i] = estimate_opt_dG1

fig,axes,cbar_ax = make_contours(dG1_mesh,dGT_mesh,log10v_arr)
for i,S in enumerate(S_list):
    if S > 0.1:
        S = int(S)
    label = f"[S]={S}"
    axes[i].plot(estimate_opt_dG1_arr[i], dGT_mesh[:,0], "k--", lw = 2)
    axes[i].text(19,-34,label,ha="right",fontsize = 18)
plt.savefig("v_vs_dG1_dGT.pdf")
plt.show()    
#%%###############################
### Fig.5 ########################
##################################
# Calculation using the most basic assumptions, ie all alphas = 0.5, k10 = k20 = 1

# Parameters for the volcano plot (Fig. 4)
Km_arr = np.logspace(-2.5,2.5)
dG1_arr = get_dG1_from_Km(Km_arr, dGT)
c_list = [k,b,g,r]
fig =plt.figure(figsize = (8,6))
ax = fig.add_axes([0.2,0.2,0.5,0.7])
for i,S in enumerate(S_list):
    v = get_Km_v(dG1_arr,dGT,S)[1]
    log10v = np.log10(v)
    if S > 0.1:
        S = int(S)
    label = f"[S]={S}"
    ax.plot(Km_arr, log10v, c= c_list[i], lw = 3, label = label)
    ax.axvline(S, c = c_list[i], ls = "--", lw =1)
ax.legend(loc = "upper left")    
ax.set_xlim(Km_arr.min(),Km_arr.max())
ax.set_xscale("log")
ax.set_xticks([1E-2, 1, 1E+2])
ax.tick_params(axis='x', which='major', pad=8)
ax.set_xlabel("$K_\mathrm{m}$ [$\mu$M]")
ax.set_ylabel("log$_{10} v$ [$\mu$M/s]")
plt.savefig("Volcano_S_dependence.pdf", dpi = 600)
plt.show()
  
#%%###############################
### Fig.6 ########################
##################################
# Parameters for the contour plots involving deviations from MM
# No effect of allostericity can be observed at S = 1 (S**n =1 for any n), therefore set a different value of S as S_dev 
S_dev = 10 # 
v_arr = np.zeros((4,arr_size,arr_size))
estimate_opt_Km_arr = np.zeros((4,arr_size))
true_opt_Km_arr = np.zeros((4,arr_size))
estimate_opt_dG1_arr = np.zeros((4,arr_size))
true_opt_dG1_arr = np.zeros((4,arr_size))
Km, v_arr[0], estimate_opt_Km_arr[0], true_opt_Km, estimate_opt_dG1_arr[0], true_opt_dG1_arr[0]  = get_Km_v_reverse(dG1_mesh,dGT_mesh,S_dev,10.001)
Km, v_arr[1], estimate_opt_Km_arr[1], true_opt_Km, estimate_opt_dG1_arr[1], true_opt_dG1_arr[1]  = get_Km_v_inhibition(dG1_mesh,dGT_mesh,S_dev,10,1)
Km, v_arr[2], estimate_opt_Km_arr[2], true_opt_Km, estimate_opt_dG1_arr[2], true_opt_dG1_arr[2]  = get_Km_v_inhibition(dG1_mesh,dGT_mesh,S_dev,1,10)
Km, v_arr[3], estimate_opt_Km_arr[3], true_opt_Km, estimate_opt_dG1_arr[3], true_opt_dG1_arr[3]  = get_Km_v(dG1_mesh,dGT_mesh,S_dev,a1 = 0.2, a2=0.2)
log10v_arr = np.log10(v_arr)
fig,axes,cbar_ax = make_contours(dG1_mesh,dGT_mesh,log10v_arr)

for i in range(4):
    s = np.log10(S)
    axes[i].plot(estimate_opt_dG1_arr[i], dGT_mesh[:,0], "k--", lw =2)
    axes[i].plot(true_opt_dG1_arr[i],dGT_mesh[:,0], "k-", lw=2)
    if i ==0:
        axes[i].text(3,10,"$K_\mathrm{m}$ = [S]",ha="right", fontsize = 14)
        axes[i].text(19,-35,"$K_\mathrm{m}$ = [S]\n$ + K g_T$ [P]",ha="right", fontsize = 14)

    if i == 1:
        axes[i].text(19,-15,"$K_\mathrm{m}$ = [S]",ha="right", fontsize = 14)
        axes[i].text(-18,10,r"$K_\mathrm{m} = \frac{[\mathrm{S}]}{1+ \gamma}$",ha="left", fontsize = 14)    
    if i ==2:
        axes[i].text(3,10,"$K_\mathrm{m}$ = [S]",ha="right", fontsize = 14)
        axes[i].text(17,-30, "$K_\mathrm{m}$ = \n" r"(1+ $\gamma$)[S]",ha="right", fontsize = 14)

    if i ==3:
        axes[i].text(19,-12,"$K_\mathrm{m}$ = [S]",ha="right", fontsize = 14)
plt.savefig("v_vs_dG1_dGT_inhibition_alpha.pdf")
plt.show()    

#%%#######################################
### Fig. 7 ###############################
##########################################
threshold_list = [0,50,300,999]
label_list = ["<50", "$\geq$50", "$\geq$300"] 
c_list = [r,lb,grey]


classified_Km_S_data = classify_Km_S_data(threshold_list)
L = len(classified_Km_S_data)

fig = plt.figure(figsize = (8,4))
ax1 = fig.add_axes([0.12,0.2,0.3,0.6])
x = np.linspace(-8,0)
ax1.fill_between(x, x-1, x+1, color ="grey", alpha = 0.3)
for i in range(L):
    logKm,logM = classified_Km_S_data[i]
    diff = sorted(logKm - logM)
    mu, s, pdf = fit_Gaussian(diff)
    ax1.scatter(logM, logKm, c = c_list[i], s = 3, label = label_list[i])

    ax = fig.add_axes([0.6,0.2+0.2*(2-i),0.3,0.2])
    ax.hist(diff, bins = 50, range = [-4,4], lw = 1, edgecolor = "k", fc = c_list[i])
    ax2 = ax.twinx()
    ax.axvline(mu, c = c_list[i], ls = "--")
    ax2.plot(diff, pdf, color = c_list[i], label = label_list[i])
    ax.text(0.95,0.7,label_list[i], transform=ax.transAxes, c = c_list[i], fontsize = 14, ha="right")
    ax.set_xlim(-4,4)
    ax2.set_ylim([0,0.5])
    ax2.set_yticks([])
    ax.set_ylim(0,50)
    ax.set_xticks([])
    ax.set_yticks([0,30])
    if i == 0:
        ax.set_ylim(0,100)
        ax.set_yticks([0,60])
        ax2.set_ylim([0,0.32])
        ax.text(-0.38,0.95,"b", transform=ax2.transAxes, fontsize = 20, weight = "bold")
        ax.text(-0.38, -1.5, "Counts [-]", transform=ax.transAxes, rotation = 90)
    if i == 2:        
        ax.set_xticks([-4,0,4])
        ax.set_xlabel("log $K_\mathrm{m}$/[S] [-]")    
ax1.plot([-8,0],[-8,0], "k--")
ax1.set_xlim(-8,0)
ax1.set_ylim(-8,0)
ax1.set_xticks([-8,-4,0])
ax1.set_yticks([-8,-4,0])
ax1.set_xlabel("log [S] [$\mu$M]")
ax1.set_ylabel("log $K_\mathrm{m}$ [$\mu$M]")
ax1.text(-0.34,0.95,"a", transform=ax1.transAxes, fontsize = 20, weight = "bold")
plt.savefig("Km_vs_S_bioinformatics.pdf", dpi = 600)  

#%%#################################################
### FIGURES FOR THE SUPPORTING INFORMATION #########
####################################################
####################################
### Influence of delta GT ##########
####################################
dGT_list = [-80,0,40]
S_max_list = [8,30,30000]
h_list = [-90,-10,-5]
ylim_list = [55,0.025,0.00035]
for i,dGT_val in enumerate(dGT_list):    
    print(dGT_val)
    dG1_list = [dGT_val/2-5,dGT_val/2,dGT_val/2+5] # kJ/mol
    S_max = S_max_list[i]
    draw_Fig3(dG1_list, dGT=dGT_val, S=np.linspace(0,S_max,101), h=h_list[i], ylim = ylim_list[i])

#%%

####################################
### Influence of k10 and k20 #######
####################################
k10_arr,k20_arr = np.meshgrid([0.1,10],[0.1,10])
v_arr = np.zeros((4,arr_size,arr_size))
Km_arr =np.zeros((4,arr_size,arr_size))
for i in range(4):
    k10 = k10_arr.flatten()[i]
    k20 = k20_arr.flatten()[i]
    Km, v, estimate_opt_Km, true_opt_Km, estimate_opt_dG1, true_opt_dG1 = get_Km_v(dG1_mesh,dGT_mesh,S_dev,k10=k10,k20=k20)
    v_arr[i] = v
    Km_arr[i] = Km
log10v_arr = np.log10(v_arr)
fig,axes,cbar_ax = make_contours(dG1_mesh,dGT_mesh,log10v_arr)

s = np.log10(S_dev)
for i in range(4):
    axes[i].contour(dG1_mesh,dGT_mesh, np.log10(Km_arr[i]), colors = "k", levels = [s], linestyles = "--", linewidths = 2)
plt.savefig("v_vs_k10_k20.pdf", dpi = 600)
plt.show()    


#%%####################################
### Influence of Inhibition ###########
#######################################
scale = 2
a = np.logspace(-scale,scale) # alpha of inhibition = [I]/Ki
def get_relv(a):
    return 2*np.sqrt(1+a)/(2+a)
x=10
y = get_relv(x)

fig = plt.figure(figsize=(6,4))
ax = fig.add_axes([0.2,0.2,0.7,0.7])
ax.plot(a,get_relv(a), c = r, ls = "--")
ax.plot(x,y,c = b, marker = "o", ms = 7)
ax.plot([a.min(),x],[y,y], c = b, ls = "--")
ax.plot([x,x],[0,y], c = b, ls = "--")
ax.set_xscale("log")
ax.set_xlim(a.min(),a.max())
ax.set_ylim(0,1.1)
ax.set_xlabel(r"$ \gamma \equiv \mathrm{[I]}/K_i$")
ax.set_ylabel(r"$ \phi \equiv \frac{v_{K_m=\mathrm{[S]}}}{v_{K_m=\mathrm{[S]}'}}$")
ax.tick_params(axis='x', which='major', pad=8)
plt.savefig("relvS_vs_alpha.pdf", dpi = 600)
#%%####################################
### Allostericity #####################
#######################################
n_allo_list = [2,3,-0.5,-1]
v_arr = np.zeros((4,arr_size,arr_size))
opt_dG1_arr = np.zeros((4,arr_size))
pred_opt_dG1_arr = np.zeros((4,arr_size))

for i in range(4):
    Km, v_arr[i], opt_Km, opt_dG1_arr[i], pred_opt_dG1_arr[i]  = get_Km_v_allosteric(dG1_mesh,dGT_mesh,S_dev,n_allo_list[i])
log10v_arr = np.log10(v_arr)
fig,axes,cbar_ax = make_contours(dG1_mesh,dGT_mesh,log10v_arr)
for i in range(4):
    s = np.log10(S_dev)
    axes[i].plot(pred_opt_dG1_arr[i], dGT_mesh[:,0], "k--", lw =2)
    axes[i].plot(opt_dG1_arr[i],dGT_mesh[:,0], "k-", lw=2)
    axes[i].text(19,-38, f"$n = {n_allo_list[i]}$", ha="right", fontsize = 14)
plt.savefig("v_vs_dG1_dGT_allosteric.pdf")
plt.show()        

#%%####################################
### Influence of alpha1 and alpha2 ####
#######################################
a1,a2 = np.meshgrid([0.3,0.7],[0.3,0.7])
a1 = a1.flatten()
a2 = a2.flatten()
log10v_arr = np.zeros((4,arr_size,arr_size)) # 4 panels
for i in range(4):
    Km, v, estimate_opt_Km, true_opt_Km, estimate_opt_dG1, true_opt_dG1= get_Km_v(dG1_mesh,dGT_mesh,S_dev, a1 = a1[i],a2=a2[i])
    log10v_arr[i] = np.log10(v)
fig,axes,cbar_ax = make_contours(dG1_mesh,dGT_mesh,log10v_arr)

# add minor annotations
for i in range(4):
    label = fr"({a1[i]},{a2[i]})"
    axes[i].contour(dG1_mesh,dGT_mesh, Km, colors = "k", levels = [S], linestyles = "--", linewidths = 2)
    axes[i].text(19,-35,label,ha="right",fontsize = 14)
plt.savefig("v_vs_dG1_dGT_a1_a2.pdf")
plt.show()    