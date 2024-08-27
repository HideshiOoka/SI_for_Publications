#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:39:29 2022

@author: Hideshi_Ooka
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmasher as cmr
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
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
plt.rcParams['lines.markersize'] =5
plt.rcParams['savefig.dpi'] = 600
blue, dblue,lblue,gblue = ["#0C7185", "#033453","#0DAFE1","#4C837A"]
red,orange, pink =["#910C07","#F48153", "#FF5657"]
c_list = [dblue,lblue,orange,red]
fig_dir = "../Figures/"
RdBu_r_short = cmr.get_sub_cmap('RdBu_r', 0.1, 0.9)

####################################
### Scheme 1 was made using PPT ####
####################################
#%%#################################
### Visualization of the Theory ####
####################################

NA = 6.02 * 10**23 # [1/mol]
N = 10000*1E14 / NA # [mol]
"""
If there is 1 site per 1 nm^2 geometric area, then there would be 1E14 sites per cm2.  In reality, there may be 10000 per 1 nm^2 geometric area (not unreasonable for rough electrodes). Convert that into mol/cm2 by dividing by Avogadro number. 
Assume the electrode is 1 cm^2 area (i.e., ignore the cm^2 unit)
"""

def get_j_from_Snum_tau(Snum_target = np.linspace(1,1E6,101), # unitless
                        tau_target = np.linspace(3600*1000,0.1, 101)): # [s]
    Snum, tau = np.meshgrid(Snum_target, tau_target) # Snum [-], tau [s]
    vOER = N / tau * Snum # [mol/s]
    jOER = vOER*4*96485*1000 # conversion to mA
    logjOER = np.log10(jOER)
    Snum_rescaled = Snum/1E6 # rescaled for visibility
    tau_hr = tau/3600
    return Snum_rescaled, tau_hr, logjOER

def get_Snum_from_j_tau(j_target = np.linspace(0.1,1000,101), # mA/cm2
                        tau_target = np.linspace(3600*1000,0.1, 101)): # [s]
    jOER, tau = np.meshgrid(j_target, tau_target) # up to 1000 mA/cm2, 1000 hrs
    tau_hr = tau/3600 # rescaled from units to hour scale (3600 sec/hr)
    vOER = jOER/4/96485/1000 # [mol/s]
    Snum = tau * vOER /N
    logSnum = np.log10(Snum)
    return jOER, tau_hr, logSnum


MIN, MAX = 0.5,4.5
contour_levels = np.arange(MIN, MAX, 0.05)
line_levels = np.arange(1, 3.6, 0.5)

x,y = 0.45, 0.3
xm,ym,xd,yd = 0.3, 0.1, 0.05, 0.2

fig = plt.figure(figsize = (8,12))
ax1 = fig.add_axes([xm,ym+y+yd,x,y])
X,Y,Z = get_j_from_Snum_tau()
im = ax1.contourf(X, Y, Z, 50, cmap = RdBu_r_short, levels = contour_levels, extend='both')
lines = ax1.contour(X, Y, Z, levels = line_levels, linewidths = 0.5, colors=[(0,0,0,0.5)])
ax1.clabel(lines, fmt='%2.1f', fontsize=14)
ax1.set_xlabel(r"$S_\mathrm{num}$ ($\times 10^6$) [-]")
ax1.set_ylabel(r"$\tau$ [hr]")
cbar_ax = fig.add_axes([xm+x+xd, ym+y+yd, 0.02, y])
cbar = fig.colorbar(im, cax=cbar_ax, ticks = np.arange(int(MIN),int(MAX)+1))
cbar_ax.set_title(r"log$_{10} j$ [mA/cm$^2$]", fontsize = 20, pad = 20)
ticklabs = cbar_ax.get_yticklabels()
cbar_ax.yaxis.set_tick_params(pad=10)

MIN, MAX = 4,7
contour_levels = np.arange(MIN, MAX, 0.05)
line_levels = np.arange(MIN+1, MAX+0.1, 0.5)
ax2 = fig.add_axes([xm,ym,x,y])
X,Y,Z = get_Snum_from_j_tau()
im2 = ax2.contourf(X, Y, Z, 50, cmap = RdBu_r_short, levels = contour_levels, extend='both')
lines = ax2.contour(X, Y, Z, levels = line_levels, linewidths = 0.5, colors=[(0,0,0,0.5)])
ax2.clabel(lines, fmt='%2.1f', fontsize=14)
ax2.set_xlabel(r"$j_\mathrm{OER}$ [mA/cm$^2$]")
ax2.set_ylabel(r"$\tau$ [hr]")
cbar_ax = fig.add_axes([xm+x+xd, ym, 0.02, y])
cbar = fig.colorbar(im2, cax=cbar_ax, ticks = np.arange(int(MIN),int(MAX)+1))
cbar_ax.set_title(r"log$_{10} S_\mathrm{num}$ [-]", fontsize = 20, pad = 20)
ticklabs = cbar_ax.get_yticklabels()
cbar_ax.yaxis.set_tick_params(pad=10)

ax1.text(-0.15,1.1,"A", transform=ax1.transAxes, fontsize = 28, weight = "bold", color = "k")
ax2.text(-0.15,1.1,"B", transform=ax2.transAxes, fontsize = 28, weight = "bold", color = "k")
plt.savefig(f"{fig_dir}Simulations.png")



#%%

def SG_filter(x):
    return savgol_filter(x, 11, 1)
##############
max_t_list =[60,50,30]
ls_list =[dblue,red,orange]
label_list = ["Mn$^{3+}$", "MnO$_4^-$"]
insitu_dir = "Experimental_Data/insitu_UVVis/"
date_ID_list = ["210913_210715A", #50 mA
                "210909_210130C", #65 mA
                "211108_211029A", #85 mA 
                "221111_221111A", #50 mA Pi
                "211028_210916B", #65 mA Pi
                "211105_210715D"] #85 mA Pi 

x,y,w,h,hsp,vsp = 0.08, 0.15, 0.15, 0.325,0.18,0.125
t_max_list = [65,44,34]
fig = plt.figure(figsize =(16,12))
for i, date_ID in enumerate(date_ID_list):
    UV_file = date_ID + "_UV.csv"
    CA_file = date_ID + "_CA.csv"
    ax1 = fig.add_axes([x + (w+hsp)*(i%3), y + (h+vsp)*(1-i//3)+h*0.55, w, h*0.45])
    ax2 = fig.add_axes([x + (w+hsp)*(i%3), y + (h+vsp)*(1-i//3), w, h*0.45])
    cbar_ax = fig.add_axes([x + (w+hsp)*(i%3)+w*1.1, y + (h+vsp)*(1-i//3), w*0.05, h*0.45])
    
    df_CA = pd.read_csv(insitu_dir + CA_file)
    t_CA,j_CA = df_CA.values.T
    fitted_file = UV_file.replace("UV","Fitted")
    df_fitted = pd.read_csv(insitu_dir + fitted_file, index_col = 0)

    III_H2SO4, VII, background, III_H3PO4, r2 = df_fitted.values.T # epsilon was Abs/M/cm units
    III = III_H2SO4 + III_H3PO4
    III *=1E6 / 8.5 # 1E6 for the M to microM conversion. 8.5 is the optical path considering 8 cm glass cell + 2 silicon rubbers
    VII *=1E6 / 8.5 # 1E6 for the M to microM conversion. 8.5 is the optical path considering 8 cm glass cell + 2 silicon rubbers
    df_UV = pd.read_csv(insitu_dir+UV_file, index_col = 0) 
    df_UV -= background # subtract background   
    UV_SG = df_UV.apply(SG_filter).values
    t = df_UV.columns.astype(float)
    wavelength = df_UV.index.astype(float)
    t_mesh, wavelength_mesh = np.meshgrid(t, wavelength)

    ax1.plot(t_CA,j_CA, "k", label = "j")
    ax_C = ax1.twinx()
    for j in range(2):
        ax_C.plot(t, [III,VII][j], ls_list[j], label = label_list[j])
    ax1.set_ylim(0,100)
    ax_C.set_ylim(-1,50)
    ax1.set_xlim(np.min(t),np.max(t))
    ax1.set_ylabel("$j$ [mAcm$^{-2}$]")
    ax_C.set_ylabel("Conc [$\\mu$M]")
    ax1.set_xticklabels([])
    n = 100
    vmin, vmax = 0, 0.5
    levels = np.linspace(vmin, vmax, n+1)
    im = ax2.contourf(t_mesh, wavelength_mesh, UV_SG, levels = levels, cmap = "RdBu_r", extend='both')
    ax2.set_ylim(800,280)
    ax2.set_yticks(np.arange(800,280,-200))
    ax2.set_ylabel("wavelength [nm]")
    ax2.set_xlabel("time [hr]")
    cbar = fig.colorbar(im, cax=cbar_ax, ticks = [0,0.2,0.4], label = "Abs [-]")
    ax1.text(-0.4,1.1,"ABCDEF"[i], transform=ax1.transAxes, fontsize = 28, weight = "bold", color = "k")
    ax1.set_xticks([0,20,40,60])
    ax2.set_xticks([0,20,40,60])
    t_max = t_max_list[i%3]
    ax1.set_xlim(0,t_max)
    ax2.set_xlim(0,t_max)
plt.savefig(f"{fig_dir}Timecourse.png")



#%%##################################################
### COMPARISON BETWEEN THEORY AND EXPERIMENTS #######
#####################################################
c1, c2, c3 = blue, red, orange
df = pd.read_csv("Experimental_Data/MnO2.csv", index_col = 0)
summary = pd.read_csv("Experimental_Data/Summary_Experimental_Conditions.csv", index_col = 1)
# has_uv = ["txt" in s for s in summary.UV.astype(str)]
# summary = summary[has_uv]


def make_parity_plot_log(df, evaluation_percentage, fig=None, ax1 = None, ax2 =None, c = c1, n_bins = 101, annotate = True):
    vd, jOER, tau10,tau37,tau50,tau90 = df.values.T
    x = -np.log(vd) - np.log(jOER)
    x = x.reshape(-1,1)
    if evaluation_percentage == 10:
        y = tau10
    elif evaluation_percentage == 37:
        y = tau37
    elif evaluation_percentage == 50:
        y = tau50
    elif evaluation_percentage == 90:
        y = tau90
    y = np.log(y)
    if fig == None:
        fig = plt.figure(figsize = (6,6))
    if ax1 == None:
        ax1 = fig.add_axes([0.15,0.2,0.6,0.6])
        # ax1.fill_between(X, Y_theory*0.8, Y_theory*1.2, color = c, alpha = 0.1)
    
    ax1.plot(x,y, "x", c = c, ms = 5)
    ax1.set_xlabel(r"$- \log v_\mathrm{d} - \log j_\mathrm{OER}$")
    ax1.set_ylabel(r"$\log \tau$ [hr]")
    
    reg=LinearRegression().fit(x,y)
    a, b = reg.coef_[0], reg.intercept_
    y_theory = reg.predict(x)
    mae = np.mean(np.abs(y- y_theory))
    r2 = 1 - np.sum((y-y_theory)**2)/np.sum((y - np.mean(y))**2)
    print(mae,r2)
    
    X = np.linspace(0, 10).reshape(-1,1)
    Y_theory = reg.predict(X)
    ax1.plot(X, Y_theory, c3, ls = "--", lw = 1)
    if annotate == True:
        ax1.text(6.2, 4.8, f"$y = ax+b$\n$a$ = {a:0.2f}\n$b$ = {b:0.2f}\n$r^2$={r2:0.2f}", fontsize = 14, va = "top")

    ax1.set_xlim(6,9)  
    ax1.set_ylim(2.5,5)
    ax1.set_yticks(np.arange(3,5.1))
    return fig, ax1, ax2

################################################
### Fig. 4 Data without pumping or Mn7 #########
################################################
not_pumped = ["pump" not in s for s in summary.action.astype(str)]
no_MnVII =  ["Mn7" not in s for s in summary.action.astype(str)]
filter = [a and b for a, b in zip(not_pumped, no_MnVII)]
for evaluation_percentage in [10, 37, 50, 90]:
    fig, ax1, ax2 = make_parity_plot_log(df[filter], evaluation_percentage)
    plt.savefig(f"{fig_dir}Parity_Plot_{evaluation_percentage}.png",dpi = 600)
#%%#############################################
### Fig. S3 Dependence of Lifetime Criteria r ##
################################################
not_pumped = ["pump" not in s for s in summary.action.astype(str)]
no_MnVII =  ["Mn7" not in s for s in summary.action.astype(str)]
filter = [a and b for a, b in zip(not_pumped, no_MnVII)]

c_list = [blue, orange, red]
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_axes([0.15,0.2,0.6,0.6])
for i, evaluation_percentage in enumerate([10, 90]):
    fig, ax1, ax2 = make_parity_plot_log(df[filter], evaluation_percentage, c = c_list[i], fig = fig, ax1 = ax1, annotate = False)
plt.savefig(f"{fig_dir}Parity_Plot_r_Dependence.png",dpi = 600)


#%%#############################################
### Fig. S4 Data with Mn7 ######################
################################################
not_pumped = ["pump" not in s for s in summary.action.astype(str)]
MnVII =  ["Mn7" in s for s in summary.action.astype(str)]
filter = [a and b for a, b in zip(not_pumped, MnVII)]
fig, ax1, ax2 = make_parity_plot_log(df[filter], 37, c = red)
plt.savefig(f"{fig_dir}Parity_Plot_VII.png",dpi = 600)

#%%####################################
######### FIGURES FOR THE SI ##########
#######################################
######### Characterization   ##########
#######################################
annotation = [f"{time} hrs" for time in [0,8,24,40]]
x_mins =  [20,300]
x_maxs =  [80,1000]
offsets = [600,0.2]
ref_c = [lblue,orange]
w,hsp = 0.35,0.1
c_list = [blue, lblue, orange, red]

fig = plt.figure(figsize = (8,4))
for m, measurement in enumerate(["XRD","Raman"]):
    df = pd.read_csv(f"Experimental_Data/Characterization/{measurement}.csv")
    N = df.shape[1]//2
    ax = fig.add_axes([0.1+(w+hsp)*m,0.2,w,0.7])
    for i in range(N):
        x,y = df.iloc[:,2*i:2*(i+1)].values.T
        ax.plot(x,y + i*offsets[m], c = c_list[i])
        ax.text(x_maxs[m]*0.98, y[250] + offsets[m]*i-90*(1-m), annotation[i] , ha = "right", va ="top", c = c_list[i], fontsize = 14)
    ax.set_ylabel("Intensity [au]")        
    ax.set_yticks([])
    ax.set_xlim(x_mins[m],x_maxs[m])
    ax.text(0.05,0.9,"AB"[m], transform=ax.transAxes, fontsize = 20, weight = "bold", color = "k")
    if m == 0:
        ax.set_ylim(-600,2800)
        ax.set_xlabel(r"2 $\theta$ [deg.]")
        for j,material in enumerate(["FTO","MnO2"]):
            ref = pd.read_csv(f"Experimental_Data/Characterization/XRD_Reference_{material}.txt",sep='\s+')
            peaks = ref["2theta"][ref["I"]>10].values
            ax.vlines(peaks, -600, -350, color = ref_c[j]) 
    else:
        ax.axvline(537, c = "k", ls = ":", alpha = 0.5)    
        ax.axvline(663, c = "k", ls = ":", alpha = 0.5)    
        ax.set_xlabel("Wavenumber [cm$^{-1}$]")
plt.savefig(f"{fig_dir}Characterization.png",dpi = 600)

#%%####################################
######### Calculation of FE  ##########
#######################################
def get_dvdt(t,v):
    dv = np.diff(v)
    dt = np.diff(t)
    dvdt = -dv/dt
    dvdt[dvdt<0] = np.nan
    return dvdt

def get_expected_dv(I):
    expected_dv = I/4/96485*22.414*3600
    return expected_dv

FE_files = ["230116_I_O2.csv"]
for file in FE_files:
    df = pd.read_csv(f"Experimental_Data/FE/{file}")

    fig = plt.figure(figsize = (8,4))
    ax = fig.add_axes([0.25,0.15,0.5,0.7])
    t_I,I,t_v,v = df.values.T
    t_I /=3600 # sec --> hrs
    I *=1000 # A --> mA
    t_v /=3600 # sec --> hrs
    v /=1 # mL --> mL
    expected_dv = get_expected_dv(I)

    dvdt = get_dvdt(t_v,v)
    ax.plot(t_I,I, c = dblue)
    ax.set_ylim(0,100)
    ax.set_ylabel("$I$ [A]")
    ax.set_xlabel("$t$ [hr]")
    ax2 = ax.twinx()
    ax2.plot(t_I,expected_dv,c=orange, lw = 1)
    ax2.scatter(t_v[:-1], dvdt, marker = "x", c = orange, s = 30)
    ax2.tick_params(axis='y', colors=orange)
    ax2.spines['right'].set_edgecolor(orange)
    ax2.set_ylim(0,22)
    ax2.set_ylabel(r"$dv/dt$ [mL/hr]", color = orange)
    figname = file.replace("csv","png")
    plt.savefig(f"{fig_dir}" + figname, dpi = 600)
