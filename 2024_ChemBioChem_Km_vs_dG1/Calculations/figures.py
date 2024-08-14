#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.integrate import solve_ivp
import cmasher as cmr
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
plt.rcParams['savefig.dpi'] = 600
blue, dblue,lblue,gblue = ["#0C7185", "#033453","#0DAFE1","#4C837A"]
red,orange, pink =["#910C07","#F48153", "#FF5657"]
RdBu_short = cmr.get_sub_cmap('RdBu', 0.1, 0.9)
RdBu_r_short = cmr.get_sub_cmap('RdBu_r', 0.1, 0.9)
fig_dir = "../Figures/"
RT = 8.314 *300/1000 # kJ/mol
ln10 = np.log(10)

#%%################################################
### This Block Draws the Free Energy Landscape ####
### of Michaelis-Menten Enzymes ###################
###################################################
scale = 20
k = "#141418"
b = "#083E73"
r = "#B10026"
g = "#228B38"
lb = "#3C93C2"
grey = "#4B5450"
def get_polynomial(x1,y1,x2,y2):
    # This will draw a 3rd order polynomial y = f(x) which passes (x1,y1) and (x2,y2) and is smooth at both ends    
    a = 6*(y2-y1)/(x1-x2)**3
    C = y1 + a/6*x1**2*(x1-3*x2)
    x = np.linspace(x1,x2)
    y = a*(x**3/3 -(x1+x2)/2*x**2 + x1*x2*x)+C
    return x,y

def get_Ea_curve(x1,y1,x2,y2, w, alpha = 0.5, y0=25):
    y_peak = y0 + y1 + alpha*(y2-y1)
    x_first_half,y_first_half = get_polynomial(x1+w,y1, (x1+x2)/2, y_peak)
    x_second_half,y_second_half = get_polynomial((x1+x2)/2, y_peak, x2-w, y2)
    x = np.append(x_first_half, x_second_half)
    y = np.append(y_first_half, y_second_half)
    return x,y

def draw_MM_mechanism(ax):
    ax.axis("off")
    ax.set_ylim(-1,0.5)
    ax.text(0,0, "E+S", ha = "center", va = "center")
    ax.text(1,0, "ES", ha = "center", va = "center")
    ax.text(2,0, "E+P", ha = "center", va = "center")
    w = 0.25
    l = 1-2*w
    y_offset = 0.15
    ax.quiver(w,y_offset, l,0, angles='xy', scale_units='xy', scale=1)
    ax.quiver(1-w,-y_offset, -l,0, angles='xy', scale_units='xy', scale=1)
    ax.quiver(1+w,0, l,0, angles='xy', scale_units='xy', scale=1)
    ax.text(0.5, y_offset*5, "$k_1$", ha = "center", va ="center", fontsize = 14)
    ax.text(0.5, -y_offset*5, "$k_{1r}$", ha = "center", va ="center", fontsize = 14)
    ax.text(1.5, y_offset*5, "$k_2$ ($k_{cat}$)", ha = "center", va ="center", fontsize = 14)

def draw_landscape(dG1, dGT, ax, c = "k", Ea0=25, w = 0.15, w_offset = 1):
    xs = np.arange(0,3)
    ys = np.array([0, dG1, dGT])
    N = len(xs)
    landscape_x = np.array([])
    landscape_y = np.array([])
    for i in range(N):
        ax.plot([xs[i]-w, xs[i]+w], [ys[i],ys[i]], c = "k", lw = 3, zorder = 99)
        try:
            x,y = get_Ea_curve(xs[i], ys[i], xs[i+1], ys[i+1], w = w)
            landscape_x = np.append(landscape_x,x)
            landscape_y = np.append(landscape_y,y)
        except IndexError:
            pass
    ax.plot(landscape_x,landscape_y, c=blue, lw = 2)
    h = dGT*1.1
    ax.text(0,h, "E+S", ha = "center", va = "top")
    ax.text(1,h, "ES", ha = "center", va = "top")
    ax.text(2,h, "E+P", ha = "center", va = "top")
    ax.set_ylabel("Gibbs Free Energy")
    ax.set_xticks([])
    ax.set_xlim(-0.5,2.5)
    ### Annotations here
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    for y in ys:
        ax.plot([-0.5, 2.5], [y,y], "--", c = "grey", lw = 1, zorder = 0)    
    dG_labels = [r"$\Delta G_1$", r"$\Delta G_2$", r"$\Delta G_\mathrm{rxn}$"]
    for i in range(N):
        y1 = ys[i%N]    
        y2 = ys[(i+1)%N]    
        x = xs[i]
        ax.quiver(x,y1,0,y2-y1, angles='xy', scale_units='xy', scale=1, color = red)
        ax.quiver(x,y2,0,y1-y2, angles='xy', scale_units='xy', scale=1, color = red)    
        ax.text(x+w/10, (y1+y2)/2, dG_labels[i], c = red, fontsize = 18, va = "center") 

def draw_scheme(dG1, dGT, S, ylim = 1):
    c = "k"
    ha_list = ["center","left","center"]
    fig = plt.figure(figsize = (6,4))
    ax1 = fig.add_axes([0.13, 0.85,0.7,0.1]) 
    ax2 = fig.add_axes([0.13, 0.1,0.7,0.7], sharex = ax1) 
    draw_MM_mechanism(ax1)
    draw_landscape(dG1, dGT, ax2, c)    
    plt.savefig(f"{fig_dir}MM_Scheme.png")
    plt.show()
dGT = -2*scale # kJ/mol
Smax=8
S_lin = np.linspace(0,Smax,101)
draw_scheme(-25, dGT, S_lin)

#%%######################################
### This Block Plots the Relationship ###
### Between Km, K, and dG1 ##############
#########################################
logK = np.linspace(-5,5, 201)
K = 10**logK
def get_Km(K, dG1):
    Km = np.exp(dG1/RT) * (1+K)
    return Km

c_list = [dblue, lblue, gblue, orange, red]
fig = plt.figure(figsize = (8,6))
ax = fig.add_axes([0.2,0.15,0.6,0.8])
dG1_list = np.arange(-47,3, 10).astype(int)
for i,dG1 in enumerate(dG1_list):
    Km = get_Km(K, dG1)
    logKm = np.log10(Km)
    logg1 = dG1/RT/np.log(10)
    ax.plot(logKm, logK, c=c_list[i], lw = 2, label = rf"$\Delta G_1$ = {dG1}")
    ax.axvline(logKm.min(), c="k", ls = "dotted", lw = 1)
    ax.plot(logK+logg1, logK, c = "k", ls = "dotted", lw = 1)
ax.set_xlim(-9,0)
ax.set_xticks(np.arange(-9, 0.1, 3))
ax.set_ylim(-4.5,4.5)
ax.set_xlabel(r"log$_{10}$ $K_\mathrm{m}$ [M]")
ax.set_ylabel("log$_{10}$ $K$ [-]")
ax.legend(loc = "upper left")
ax.text(-6.7,4.3, r"$\Delta G_1 = RT$ ln ($K_\mathrm{m}$ / $K$)", va = "top", rotation = 45)
ax.text(-7,-4.1, r"$\Delta G_1 = RT$ ln $K_\mathrm{m}$", rotation = 90)
plt.savefig(f"{fig_dir}K_vs_Km.png")


#%%#######################################
### Plot delta vs KA and KB###############
##########################################
def get_delta(KA,KB):
    return -RT*np.log((1+KA)/(1+KB)) # The 1000 is for J --> kJ conversion

logK_range = np.linspace(-2,2,101)
logKA, logKB = np.meshgrid(logK_range, logK_range)
KA = 10**logKA
KB = 10**logKB
delta = get_delta(KA,KB)
abs_delta = np.abs(delta)
MAX = int(delta.max()*1.1)
MIN = 0
contour_levels = np.linspace(MIN, MAX, 101)
line_levels = np.linspace(MIN, MAX, 7)

fig = plt.figure(figsize = (8,6))
ax = fig.add_axes([0.2,0.2,0.525,0.7])
im = ax.contourf(logKA, logKB, abs_delta, 50, cmap = RdBu_r_short, levels = contour_levels, extend='both')
lines = ax.contour(logKA, logKB, abs_delta, levels = line_levels, linewidths = 0.5, colors=[(0,0,0,0.5)])
ax.set_xlabel(r"log$_{10}$ $K_\mathrm{A}$")
ax.set_ylabel(r"log$_{10}$ $K_\mathrm{B}$")
ax.clabel(lines, fmt='%2.1f', fontsize=14)
cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax, ticks = line_levels)
cbar_ax.set_title(r"|$\delta$| [kJ/mol]", fontsize = 20, pad = 20)
ticklabs = cbar_ax.get_yticklabels()
cbar_ax.yaxis.set_tick_params(pad=10)
plt.savefig(f"{fig_dir}delta_vs_KA_KB.png")

#%%##################################
### Plot of Km, dGrxn, and dG1 ######
#####################################
scale = 3
dGT = np.linspace(-2*scale*RT*ln10,2*scale*RT*ln10,20001)
sgT = np.exp(dGT/RT/2)
fig = plt.figure(figsize = (8,6))
ax = fig.add_axes([0.2,0.15,0.6,0.8])
for i,dG1 in enumerate([-20,-10,0,10]):
    # g1 = np.exp(dG1/RT)
    # logg1 = np.log10(g1)
    logg1 = dG1/RT/ln10
    logKm = logg1 + np.log10(1+1/sgT)
    ax.axvline(logg1, c="k", ls = "dotted", lw = 1)
    ax.plot(((dG1-dGT/2)/RT/ln10), dGT, "k--", lw = 1)
    ax.plot(logKm, dGT, label = dG1, c = c_list[i], lw = 2)
ax.set_xlim(-3,3)
ax.set_ylim(dGT.min(),dGT.max())
ax.set_xlabel(r"log$_{10}$ $K_\mathrm{m}$ [M]") 
ax.set_ylabel(r"$\Delta G_\mathrm{rxn}$ [kJ/mol]")
ax.legend()
ax.text(0.1,30, r"$\Delta G_1 = RT ln K_\mathrm{m}$", va = "top", fontsize = 14, rotation = -90)
ax.text(0.7,-25, r"$\Delta G_1 = RT ln K_\mathrm{m} + \frac{\Delta G_T}{2}$", fontsize = 14, rotation = -45)
plt.savefig(f"{fig_dir}dGT_vs_Km.png")



#%%###################################################
### Determining delta G1 from the time trajectory ####
######################################################
# Choose K and then determine k1 and k1r
def dxdt(t,x,k1,k1r):
    S,E,ES,P = x
    v1 = k1*S*E
    v1r = k1r*ES
    v2 = k2*ES
    return np.array([-v1+v1r, -v1+v1r+v2, v1-v1r-v2,v2])

def get_k1_k1r(K): # treat Km and k2 as global parameters
    k1r = k2/K
    k1 = (k1r+k2)/Km
    return k1,k1r

def draw_triangles():
    dt = 0.3 # sec
    dt_min = dt/60
    Vss = k2*S0*E0/(Km+S0) # this is standard units M/s
    dy = Vss*dt *1000 # change from M to mM units 
    x0, y0 = 0.001, 0.9992 # minutes, mM
    x1, y1 = x0,          y0 + dy
    x2, y2 = x0 + dt_min, y0
        
    dt_min = 10
    dt = dt_min*60
    dY = Vss*dt *1000 # change from M to mM units 
    X0, Y0 = 3, 0.55 # minutes, mM # 
    X1, Y1 = X0,        Y0 + dY
    X2, Y2 = X0 + dt_min,   Y0

    ax2.plot([x1,x2],[y1,y2], "k")
    ax2.plot([x0,x1],[y0,y1], "k")
    ax2.plot([x0,x2],[y0,y2], "k")
    ax.plot([X1,X2],[Y1,Y2], "k")
    ax.plot([X0,X1],[Y0,Y1], "k")
    ax.plot([X0,X2],[Y0,Y2], "k")

x0 = np.array([1E-3, 1E-6,0,0]) # S_0 = 1mM, E_0 = 1 microM]
S0, E0 = x0[:2]
t_max = 7200 # sec units, so equivalent to 2 hrs
t_span = [0,t_max]
k2 = 1 # 1/s units
Km = 1E-3 # 1E-3 in M units is 1 mM
K_list = [0.01, 1, 100] # dimensionless units
c_list = [red, "k", lblue]

ls_list = ["o-","-", "+"]
fig = plt.figure(figsize = (8,6))
ax = fig.add_axes([0.2,0.2,0.7,0.7])
ax2 = fig.add_axes([0.55,0.5,0.3,0.35])
for i,K in enumerate(K_list):
    k1,k1r = get_k1_k1r(K)
    soln = solve_ivp(dxdt, t_span,x0, method="LSODA", t_eval = np.linspace(0, t_max, 1000001), args=(k1,k1r))
    t = soln.t
    x = soln.y
    t_min = t/60
    t_max_min = t_max/60
    S_mM = x[0] * 1000 # mM units
    S0_mM = S_mM[0]
    ax.plot(t_min[::10000],S_mM[::10000],ls_list[i], c = c_list[i], ms = 5, lw = 2)
    ax2.plot(t_min,S_mM,ls_list[i], c = c_list[i], ms = 5)
    ax2.plot(t_min, S0_mM*(1-k1*E0*t), c = c_list[i], ls = "--", lw = 1, zorder = 0) # k1*E0*t is in standard units because it will be dimensionless in the end 1/M/s *M*s
    if i == 0:
        draw_triangles()
ax.set_xlabel("$t$ [min]")    
ax.set_ylabel(r"[S] mM")
ax.set_xlim(-1,t_max_min)
ax2.set_xlabel("$t$ [min]")    
ax2.set_ylabel(r"[S] mM")
ax2.set_xlim(-0.001,0.01)
ax2.set_ylim(0.999,1)
plt.savefig(f"{fig_dir}Transient.png")



#%%############################
### Fit Cellulase Data ########
###############################
from scipy import stats
df = pd.read_csv("cellulases_pars_0.7.csv", index_col = 0)
# The original csv data was generated in Chiba, Ooka et al 2024 Angew. The raw experimental data comes from Kari et al, 2021 Nat. Commun.
Km, kcat, alpha, logC, logC_fit, C, C_fit = df.values.T
logKm = np.log10(Km)
logkcat = np.log10(kcat)
alpha = alpha[0] # only 1 value
logC_fit = logC_fit[0] # only 1 value
delta = logkcat - alpha*logKm - logC_fit
delta = sorted(delta)
mu,s = stats.norm.fit(delta)
x = np.linspace(-0.5,0.5, 201)
pdf = stats.norm.pdf(x, loc=mu, scale=s)

fig = plt.figure(figsize = (8,6))
ax = fig.add_axes([0.2,0.2,0.525,0.7])
ax.hist(delta, bins = 25, range = [-0.5,0.5], lw = 1, edgecolor = "k", fc = blue)
ax.axvline(mu, c = blue, ls = "--")
ax.set_xlim(-0.5,0.5)
ax.set_xticks([-0.5,0,0.5])
ax.set_ylim(0,20)
ax.set_xlabel(r"$\delta$ [-]")
ax.set_ylabel("Counts [-]")
ax2 = ax.twinx()
ax2.plot(x, pdf, blue)
ax2.set_ylim(-0.1,3.5)
ax2.set_yticks(np.arange(0,3.5))
plt.savefig(f"{fig_dir}Histogram.png")
# (delta/alpha).min() # -0.318
# (delta/alpha).max() # 0.360
# print(RT*(0.360+0.318))


#%%#################################
### TABLE OF CONTENTS GRAPHIC ######
####################################
logKm_range = np.linspace(-9,0, 201)
logK_range = np.linspace(-4.5,4.5, 201)
Km_range = 10**(logKm_range)
K_range = 10**(logK_range)
Km, K = np.meshgrid(Km_range, K_range)
dG1 = RT*(np.log(Km) - np.log(1+K))
logKm = np.log10(Km)
logK = np.log10(K)

MIN, MAX = -80, 0
contour_levels = np.linspace(MIN, MAX, 21)
fig = plt.figure(figsize = (5.5,5))
ax = fig.add_axes([0.17,0.15,0.6,0.66])
cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.66])
im = ax.contourf(logKm,logK, dG1, cmap = RdBu_short, levels = contour_levels)
ax.set_xlim(-9,0)
ax.set_xticks(np.arange(-9, 0.1, 3))
ax.set_ylim(-4.5,4.5)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x+3:0.0f}'))
ax.set_xlabel(r"log$_{10}$ $K_\mathrm{m}$ [mM]")  # changed to microM units
ax.set_ylabel("log$_{10}$ $K$ [-]")
cbar = fig.colorbar(im, cax=cbar_ax, ticks = np.arange(MIN, MAX+1, 20))
cbar_ax.set_title(r"$\Delta G_1$ [kJ/mol]", fontsize = 18, pad =10)
ticklabs = cbar_ax.get_yticklabels()
cbar_ax.yaxis.set_tick_params(pad=10)
ax.text(-4.5, 2.5, "Van Slyke-Cullen Region\n$\\Delta G_1 = RT$ ln ($K_\mathrm{m}$ / $K$)", ha="center", va = "center", fontsize = 16)
ax.text(-4.5, -2.5, "Michaelis-Menten Region\n$\\Delta G_1 = RT$ ln $K_\mathrm{m}$", ha="center", va = "center", fontsize = 16)
plt.savefig(f"{fig_dir}TOC.png")