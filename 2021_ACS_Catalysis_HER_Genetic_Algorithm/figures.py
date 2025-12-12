#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
import matplotlib.lines as mlines
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
plt.rcParams["xtick.minor.size"] =6.0
plt.rcParams["ytick.minor.size"] = 6.0
plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["ytick.minor.width"] = 1
plt.rcParams["legend.fontsize"] = 10
plt.rcParams['lines.markersize'] =3
plt.rcParams['savefig.dpi'] = 600
#################################################################
### Figures in the Main Text of 2021 ACS Catalysis            ###
### This file is for illustration purposes only               ###
### The main numerical analysis is independent of the figures ###
#################################################################
from HER_MK import read_CV, get_log_j_VHT, get_offset, get_log_diff
all_E, all_j_exp, all_log_j_exp, E, log_j_exp = read_CV()
#############################################
################ FIGURE 1 ###################
#############################################
fig = plt.figure(figsize=(8,4)) 
ax1 = fig.add_axes([0.2, 0.2, 0.3, 0.7])
ax1.scatter(all_E,all_j_exp,color = "r", s = 20)
ax1.set_xlim(-0.3, 0.2)
ax1.set_ylim(-1800,100)
ax1.set_xticks(np.arange(-0.2,0.21,0.2))
ax1.set_yticks(np.arange(0,-1501,-500))
ax1.set_xlabel("$E - iR$ [V vs. RHE]")  
ax1.set_ylabel("$j$  [mA cm$^{-2}$]") 

ax2 = fig.add_axes([0.65, 0.2, 0.3, 0.7]) 
ax2.scatter(all_E,all_log_j_exp, color = "r", s = 20)
ax2.plot(all_E,all_log_j_exp, color = "r")
ax2.set_xlim(-0.3, 0.2)
ax2.set_ylim(-2,4)
ax2.set_xticks(np.arange(-0.2,0.21,0.2))
ax2.set_yticks(np.arange(4,-2.9,-1))
ax2.set_xlabel("$E - iR$ [V vs. RHE]")  
ax2.set_ylabel("log $j$  [mA cm$^{-2}$]") 
for i, ax in enumerate([ax1,ax2]):
    ax.text(-0.35,0.95, "ABCDE"[i], transform=ax.transAxes,size=20, weight="bold")    
# plt.savefig("figures/Fig1.png")

#%%##########################################
################ FIGURE 2 ###################
#############################################
df=pd.read_csv("M_Mean_VHT_0.5.csv")
par_vec_opt = df.iloc[-1,1:-1].values # last row has the largest population the first entry is the population
GH_opt = par_vec_opt[0]
par_vec_zero = par_vec_opt.copy()
par_vec_zero[0] = 0 # hypothetical catalyst with GH = 0
log_j_VHT_opt = get_log_j_VHT(par_vec_opt)
offset_opt = get_offset(log_j_VHT_opt)
offset_log_j_VHT_opt = log_j_VHT_opt - offset_opt
offset_log_j_VHT_zero = get_log_j_VHT(par_vec_zero)- offset_opt

fig = plt.figure(figsize = (8,4))
ax = fig.add_axes([0.3, 0.2, 0.5, 0.7]) 
ax.scatter(E,log_j_exp, color = "k")
ax.plot(E,offset_log_j_VHT_opt, color = "r", linewidth = 3)
ax.plot(E,offset_log_j_VHT_zero, color = "b", linewidth = 3)
ax.set_xlabel("$E - iR$ [V vs. RHE]")
ax.set_ylabel("log $j$ [mA cm$^{-2}$]")
ax.set_xlim(-0.3,0.0)
ax.set_ylim(-0.2,3.5)
ax.set_xticks(np.arange(-0.3,0.01,0.1))
ax.set_yticks(np.arange(3,-0.1,-1))
ax.text(-0.27, 2.4, "$\Delta G_\mathrm{H}$ = " + str(GH_opt)[:5], size=20, color = "r")
ax.text(-0.27, 1.2, "$\Delta G_\mathrm{H}$ = 0", size=20, color = "b")
# plt.savefig("figures/Fig2.png")

#%%##########################################
################ FIGURE 3 ###################
#############################################

pop_dict = pd.read_csv('fitting_results/pop_dict_VHT_1000_0.5.csv', delimiter=',', header =None).values
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
for i, ax in enumerate(axes):
    im = ax.contourf(GH,logK3,log_diff, 20, cmap = "rainbow")
    j = i*10
    ax.scatter(pop_dict[j,:,0], pop_dict[j,:,2], color ="k", s=5)
    ax.text(0.05,0.85, "ABCDE"[i], transform=ax.transAxes,size=20, weight="bold")
    ax.set_xlim(-1,1)
    ax.set_ylim(-10,10)
    ax.set_xticks(np.arange(-1,1.1,1))
    ax.set_yticks(np.arange(-10,10.1,5))

# Change the color of the scatter plot to grey where it overlaps with the axis label of panel A
for j in range(pop_size): 
    GH = pop_dict[0,j,0]
    logK3 = pop_dict[0,j,2]
    if GH < -0.6 and logK3 > 6:
        ax1.scatter(pop_dict[0,j,0], pop_dict[0,j,2], color ="grey", s=5)

fig.text(x+w+sp/2, 0.05, r"$\Delta G_\mathrm{H}$ [eV]", ha = "center")  
fig.text(0.03, y+h+sp/2, "log $K_3$ [-]", va="center", rotation = "vertical")
cbar_ax = fig.add_axes([0.87, y, 0.02, 2*h+sp])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_title("log $R^2$")
ticklabs = cbar.ax.get_yticklabels()
# plt.savefig("figures/Fig3.png")

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
for i in range(4):
    if i == 0:
        scale = 0.1
    else:
        scale = 1
    ax1.plot(GH_range, scale*norm.pdf(GH_range, loc=M_Mean.iloc[-i-1,1], scale = M_STD.iloc[-i-1,1]), label = str(M_Mean.iloc[-i-1,0]), c = "r", ls = ls_list[i])
    ax2.plot(logK_range, norm.pdf(logK_range, loc=M_Mean.iloc[-i-1,2], scale = M_STD.iloc[-i-1,2]), label = str(M_Mean.iloc[-i-1,0]), c = "b", ls = ls_list[i])
    ax2.plot(logK_range, norm.pdf(logK_range, loc=M_Mean.iloc[-i-1,3], scale = M_STD.iloc[-i-1,3]), c= "g", ls = ls_list[i])
ax1.set_xlabel("$\Delta G_\mathrm{H}$ [eV]")
ax1.legend(loc = "upper left", frameon  = False)

ax2.set_xlabel("log $K$ [-]")
ax2.text(-8.5,0.7, "$K_2$",size=20, weight='bold', color = "b")
ax2.text(-5,0.6, "$K_3$",size=20, weight='bold', color = "g")
ax2.set_ylim(-0.05,1)

for i, ax in enumerate([ax1,ax2]):
    ax.text(-0.2,1, "AB"[i], transform=ax.transAxes,size=20, weight="bold")
    ax.set_ylabel("Probability Density")
# plt.savefig("figures/Fig4.png")

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
E_lin = np.linspace(-0.001, -1, 30)
for i, GH in enumerate(GH_list):
    ax1.plot([w,w+s], [0,GH], color = c_list[i])
    ax1.plot([w+s+off,2*w+s-off], [GH,GH], color = c_list[i], linewidth=5)
    ax1.plot([2*w+s,2*w+2*s], [GH,0], color = c_list[i])
    par_vec = par_vec_opt.copy()
    par_vec[0] = GH
    for j, logK in enumerate(logK_list):
        par_vec[2] = logK
        j_VT = get_log_j_VHT(par_vec, mechanism = "VT", const_kzeros = True, E = E_lin)
        ax2.plot(E_lin, j_VT, color = c_list[i], linestyle = ls_list[j])
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
# plt.savefig("figures/Fig5.png")
