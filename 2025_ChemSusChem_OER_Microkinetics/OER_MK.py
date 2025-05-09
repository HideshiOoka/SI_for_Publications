#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
########################################
### Parameters for Exporting Figures ###
########################################
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
fig_dir="../Figures/"
dblue = "#005B82"
red="#D51506"

#######################################
### Physical Constants and Settings ###
#######################################
F = 96485
R = 8.314
T = 300
f = 0.5*F/R/T # this setting implicitly assumes alpha = 0.5 for all rate constants
E_lin = np.linspace(0.6,1.85, 2001) # Potential range of OER
N = 1 # active sites mol/cm2
###########################################
### Settings for Computational Accuracy ###
###########################################

def noise_filter(v):
    mask = np.abs(v)< 1E-10
    v[mask] = np.nan
    return v

def double_check(target, ref, target_name):
    if np.allclose(target, ref):
        TF = "is"
    else:
        TF = "is not"
        print(f"{target_name} {TF} consistent")
        # move outside the else clause to display messages even when there is no problem
#%%
##################################################
### Define Catalysts Based on their Parameters ###
##################################################
TDKN = {} # thermodynamically and kinetically neutral
TDBKN = {"E1": 0.86, "E2": 1.6, "E3": 1.6} # thermodynamically biased (satisfies scaling law) and kinetically neutral
TDKB = {"E1": 0.86, "E2": 1.6, "E3": 1.6, "k10":1E-12, "k20":1E-6, "k30":1E-6, "k40":1E-12} # thermodynamically and kinetically biased = kinetically ideal
TDNKB = {"k10":1E-12, "k20":1E-6, "k30":1E-6, "k40":1E-12} # thermodynamically neutral and kinetically biased
# The TDBKN catalyst doesn't appear in the final version of the paper, but it is here for consistency

TDKN_HER = {} # thermodynamically and kinetically neutral
TDBKN_HER = {"E1": -0.2}
TDKB_HER = {"E1": -0.2, "k10":1E-7, "k20":1E-11, "k30":1E-9} # thermodynamically and kinetically biased = kinetically ideal
TDNKB_HER = {"k10":1E-7, "k20":1E-11, "k30":1E-9} # thermodynamically neutral and kinetically biased
# The HER catalysts are used for the simulations in Fig. S3-S5

##################################################
### Actual Microkinetic Simulations Start Here ###
##################################################
def get_j(E1 = 1.23,
          E2 = 1.23,
          E3 = 1.23, #E4 is defined by 4.96 - E1 - E2 - E3 
          k10=1E-9,
          k20=1E-9,
          k30=1E-9,
          k40=1E-9,
          k50 = 1E-9, 
          E=E_lin, 
          mechanism = "both"):
    E4 = 4*1.23 - (E1+E2+E3)
    eta1 = E-E1
    eta2 = E-E2
    eta3 = E-E3
    eta4 = E-E4
    eta5 = E1+E2-E3-E4
    ### The default parameter is mechanism = "both"
    k1 = k10*np.exp(f*eta1)
    k1r = k10*np.exp(-f*eta1)
    k2 = k20*np.exp(f*eta2)
    k2r = k20*np.exp(-f*eta2)
    k3 = k30*np.exp(f*eta3)
    k3r = k30*np.exp(-f*eta3)
    k4 = k40*np.exp(f*eta4)
    k4r = k40*np.exp(-f*eta4)
    k5 = k50*np.exp(f*eta5)
    k5r = k50*np.exp(-f*eta5)
    # Update parameters if mechanism is "AB"
    if mechanism == "AB":
        k5 = 0
        k5r = 0
    elif mechanism == "DC":
        k3 = 0
        k3r = 0
        k4r = 0
    epsilon = (k1*k2+k1r*k4r)*(k3+k3r) \
            + (k2*k3+k2r*k1r)*(k4+k4r) \
            + (k3*k4+k3r*k2r)*(k1+k1r) \
            + (k4*k1+k4r*k3r)*(k2+k2r)
    b = k2+k1r
    d = k4+k3r
    p = 1 + k1 /b +k4r/d
    q = 1 + k2r/b +k3 /d
    A = k5*p**2/q**2 -k5r
    B = epsilon/b/d/q + 2*k5*p*N/q**2
    C = N/q * (k5*N/q+ k1r*k2r/b+k3*k4/d)
    gamma = np.sqrt(B**2 - 4*A*C)
    X1 = 2*C/(B + gamma)
    X3 = (N-p*X1)/q
    X2 = (k1*X1+k2r*X3)/b
    X4 = (k4r*X1+k3*X3)/d

    theta1 =X1/N*100
    theta2 =X2/N*100
    theta3 =X3/N*100
    theta4 =X4/N*100

    v = k1*X1 - k1r*X2
    v_AB = k3*X3 - k3r*X4
    v_DC = k5*X3**2 - k5r*X1**2
    
    ### Section for double checking ##############

    if mechanism == "both":
        A_explicit = A
        B_explicit = B
        C_explicit = C
        B24AC_explicit = (epsilon/b/d/q)**2 + 4*k5*k5r/q**2 + 4*k5*p*(k1*k2*d+k3r*k4r*b)/b/d/q**2 + 4*k5r*(k1r*k2r*d+k3*k4*b)/b/d/q
        v_explicit = N/b/q/(B+gamma)*(2*k1*k2*k5/q -k1r*k2r*epsilon/b/d/q + 2*(k1*k2*q+k1r*k2r*p)*(k1r*k2r*d+k3*k4*b)/b/d/q - k1r*k2r*gamma)

    if mechanism == "AB":
        A_explicit = 0
        B_explicit = epsilon/b/d/q
        C_explicit = (k1r*k2r*d+k3*k4*b)/b/d/q
        B24AC_explicit = B_explicit**2
        v_explicit = (k1*k2*k3*k4-k1r*k2r*k3r*k4r)*N/epsilon
    
    if mechanism == "DC":
        epsilon_explicit = k4*(k1*k2+k1r*k2r+k1*k2r)
        double_check(epsilon_explicit, epsilon, "epsilon_DC") 
        A_explicit = A
        B_explicit = (k1*k2+k1r*k2r+k1*k2r)/b/q + 2*k5*p/q**2
        C_explicit = 1/q * (k5/q + k1r*k2r/b)
        B24AC_explicit = ((k1*k2+k1r*k2r+ k1*k2r)/b/q)**2 + 4*k5*k5r/q**2 + 4*k1*k2*k5*p/b/q**2 + 4*k1r*k2r*k5r/b/q
        v_explicit = N/b/q/(B+gamma)*(2*k1*k2*k5/q - k1r*k2r*gamma + k1r*k2r*(k1*k2 +k1r*k2r+k1*k2r)/b/q)

    j_explicit = 4*F*v_explicit*1000    
    double_check(A_explicit, A, "A")
    double_check(B_explicit, B, "B")
    double_check(C_explicit, C, "C") 
    double_check(B24AC_explicit, gamma**2, "B2-4AC")
    double_check(v_explicit, v, "v")
    double_check(v_AB+v_DC, v, "v_total and v_AB, v_DC") 
    ##############################################
    j = 4*F*v *1000 # A to mA/cm2 units
    j_AB = 4*F*v_AB*1000
    j_DC = 4*F*v_DC*1000
    logj = np.log10(np.abs(j))
    logj_AB = np.log10(np.abs(j_AB))
    logj_DC = np.log10(np.abs(j_DC))
    E_TS, TS = 0, 0
    if len(E) > 1:
        E_TS,TS, alpha_eff = get_TS(logj)
    return j, j_AB, j_DC,theta1,theta2,theta3,theta4, logj, logj_AB, logj_DC, E_TS, TS

def get_TS(logj, E = E_lin, n = 1): # n is the spacing within data points to calculate the Tafel slope (gradient)
    E = E[logj != logj.min()]
    logj = logj[logj != logj.min()]
    deltaE = E[n]-E[0]
    alpha_eff = np.gradient(logj, deltaE)*R*T*np.log(10)/F
    TS = 60/np.abs(alpha_eff)
    E_TS = E
    return E_TS, TS, alpha_eff
    
#%%##############################################
### Fig. 1 Ideal vs Nonideal Binding Energies ###
#################################################
fig = plt.figure(figsize = (8,6))
ax1 = fig.add_axes([0.15,0.15,0.7,0.5])
ax2 = fig.add_axes([0.15,0.7,0.7,0.2], sharex = ax1)
ax1.set_xlabel("$E$ [V vs. RHE]")
ax1.set_ylabel("log$_{10} |j|$ [mA/cm$^{2}$]")
ax1.set_xlim(E_lin.min(), E_lin.max())
ax1.set_ylim(-12,4)
ax2.set_ylabel("$\\theta$ [%]")
ax2.set_ylim(-5,105)
ax2.tick_params(labelbottom=False)
ax1.axvline(1.23, c = "gray", ls = ":")
ax2.axvline(1.23, c = "gray", ls = ":")
par_list = [TDKN, TDBKN]
c_list = [red, dblue]
ls_list = ["--", "-",":"]
for i, par in enumerate(par_list):
    j, j_AB, j_DC,theta1,theta2,theta3,theta4, logj, logj_AB, logj_DC, E_TS, TS = get_j(**par) # 1E-18)
    c = c_list[i]
    ax2.plot(E_lin, theta2, c, ls = "-")
    ax2.plot(E_lin, theta3, c, ls = "--")
    ax2.plot(E_lin, theta4, c, ls = ":")
    ax1.plot(E_lin,logj, c)
ax2.plot([0], [0], "k", ls = "-", label = "OH")
ax2.plot([0], [0], "k", ls = "--", label = "O")
ax2.plot([0], [0], "k", ls = ":", label = "OOH")
ax2.legend(loc = "upper left")
plt.savefig(f"{fig_dir}Ideal_CV.png")

#%%##############################################
### Fig. 2 Ideal vs Nonideal Binding Energies ###
#################################################
par_list = [TDKN, TDNKB, TDKB]
c1 = "gray"
c2 = dblue
c_list = [red, c1, c2]
for mechanism in ["both", "AB", "DC"]:
    print(mechanism)
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_axes([0.15,0.55,0.7,0.4])
    ax2 = fig.add_axes([0.15,0.1,0.7,0.4], sharex = ax1)
    ax1.set_ylabel("log$_{10} |j|$ [mA/cm$^{2}$]")
    ax1.set_ylim(-14,7)
    ax1.tick_params(labelbottom=False)
    ax1.axvline(1.23, c = "gray", ls = ":")
    ax2.axvline(1.23, c = "gray", ls = ":")
    ax2.set_ylabel("Tafel slope [mV/dec]")
    ax2.set_xlabel("$E$ [V vs. RHE]")
    ax2.set_ylim(0,150)
    ax2.set_yticks([0,40,80,120])
    ax2.axhline(30, c = "gray", ls = ":")
    ax2.axhline(40, c = "gray", ls = ":")
    ax2.axhline(60, c = "gray", ls = ":")
    ax2.axhline(120, c = "gray", ls = ":")
    ax1.text(0.05,0.9,"A", transform=ax1.transAxes, fontsize = 20, weight = "bold", color = "k")
    ax2.text(0.05,0.9,"B", transform=ax2.transAxes, fontsize = 20, weight = "bold", color = "k")
    for i, par in enumerate(par_list):
        print(par)
        j, j_AB, j_DC,theta1,theta2,theta3,theta4, logj, logj_AB, logj_DC, E_TS, TS = get_j(**par, mechanism = mechanism)
        ax1.plot(E_lin,logj, c_list[i])
        ax2.plot(E_TS, TS, c_list[i])
    plt.savefig(f"{fig_dir}BE_Comparison_{mechanism}.png")
#%%##############################################
### Fig. 3 Kinetic Volcano Plots ################
#################################################
### First, do the calculations and export the numerical values of the OER rates
log10K_scale = 8.5
E2_min, E2_max = 0.6, 1.8
num_datapoints = 1001
log10K, E2 = np.meshgrid(np.linspace(-log10K_scale,log10K_scale, num_datapoints), np.linspace(E2_min,E2_max, num_datapoints))
E3 = E2
E1 = 2.46 -E2
k10 = 1E-9 * 10**(log10K/2)
k20 = 1E-9 * 10**(-log10K/2)
k30 = k20
k40 = k10
Z_arr = np.zeros((4, num_datapoints, num_datapoints))
E_list = [1.231, 1.5, 1.7, 0.8]
for i, E in enumerate(E_list):
    E = np.array([E])
    j, j_AB, j_DC,theta1,theta2,theta3,theta4, logj, logj_AB, logj_DC, E_TS, TS = get_j(k10 = k10, k20 = k20, E1=E1, E2 = E2, E3 = E3, E = E)
    Z_arr[i] = logj
    np.savetxt(f"logj_map_{i}.csv", Z_arr[i], delimiter = ",")

### Then, plot the data ###

MIN = Z_arr.min()
MAX = 5 # Z_arr.max()
X = E2
Y = log10K
contour_levels = np.linspace(MIN,MAX,81)
fig = plt.figure(figsize=(8,6.4))
x,y,w,h,hsp,vsp = 0.15, 0.15, 0.25, 0.3125,0.1,0.125
ax1 = fig.add_axes([x,y+h+vsp,w,h]) 
ax2 = fig.add_axes([x+w+hsp,y+h+vsp,w,h]) 
ax3 = fig.add_axes([x,y,w,h]) 
ax4 = fig.add_axes([x+w+hsp,y,w,h]) 
axes = ax1,ax2,ax3,ax4
for i in range(4):
    ax = axes[i]
    ax.axhline(0, c = "white", ls = ":")
    ax.axvline(1.23, c = "white", ls = ":")
    im = ax.contourf(X,Y,Z_arr[i], 50, cmap = "jet", levels = contour_levels)
    ax.contour(X,Y,Z_arr[i], levels = contour_levels, linewidths = 0.3, colors=[(0,0,0,0.5)])
    ax.set_xticks(np.arange(E2_min,E2_max+0.1,0.4))
    ax.set_yticks(np.arange(-log10K_scale+0.5,log10K_scale+0.1,4))
    ax.text(0.05,0.8,"ABCD"[i], transform=ax.transAxes, fontsize = 20, weight = "bold")
    if i >= 2:
        ax.set_xlabel("$E_2$ [eV]")  
    if i%2 ==0:
        ax.set_ylabel("log$_{10}$ $k_1^0/k_2^0$ [-]")   
    ax.set_xlim(E2_min, E2_max)
    ax.set_xticks([0.8,1.2,1.8])
    ax.plot(1.23, 0, "wo")
    ax.text(1.23+0.03, 0 + 1, "TDKN", color = "white", fontsize = 14)
cbar_ax = fig.add_axes([x+hsp*2+2*w, y, 0.02, 2*h+vsp])
cbar = fig.colorbar(im, cax=cbar_ax, ticks = np.arange(int(MIN)+1,int(MAX)+1,5))
cbar_ax.set_title("log $|j|$ [mA/cm$^2$]", fontsize = 20, pad = 20)
ticklabs = cbar_ax.get_yticklabels()
cbar_ax.yaxis.set_tick_params(pad=10)
plt.savefig(f"{fig_dir}Volcano.png", dpi = 600)



#%%#####################################################
### Fig. S3-5 HER Ideal vs Nonideal Binding Energies ###
########################################################
E_lin_HER =  np.linspace(-0.5, 0.5, 2001)
def get_j_HER(E = E_lin_HER, E1 = 0, k10 = 1E-9, k20 = 1E-9, k30 = 1E-9, mechanism = "VHT"): 
    k1  = k10 * np.exp(f*( E1-E))        
    k1r = k10 * np.exp(f*(-E1+E))        
    k2  = k20 * np.exp(f*(-E1-E))
    k2r = k20 * np.exp(f*( E1+E))
    k3  = k30 * np.exp(f*(-2*E1))
    k3r = k30 * np.exp(f*  2*E1 )
    if mechanism == "VH":
        k3 = 0
        k3r = 0
    elif mechanism == "VT":
        k2 = 0
        k2r = 0        
    gamma = np.sqrt( (k1+k1r+k2+k2r)**2 + 4*(k3*(k1+k2r)+k3r*(k1r+k2)+k3*k3r) )
    v = 2*(k1**2*(k2+k3)-k1r**2*(k2r+k3r) +k1*k1r*(k2-k2r))/((k1+k1r)*(k1+k1r+k2+k2r+gamma)+2*(k1*k3+k1r*k3r))
    v2 = (k1*k2 - k1r*k2r)/(k1+k1r+k2+k2r)
    v3 = k1*k2/k1r
    j = - 2*F*v *1000 # A to mA/cm2 units 
    # We are going to take absolute values anyway, but strictly speaking, v is the (net) HER rate not the HOR rate, so a negative sign is necessary in the current expression. 
    logj = np.log10(np.abs(j))
    E_TS, TS = 0, 0
    if len(E) > 1:
        E_TS,TS, alpha_eff = get_TS(logj, E = E)
    return j, logj, E_TS, TS
par_list = [TDKN_HER, TDNKB_HER, TDKB_HER]
c1 = "gray"
c2 = dblue
c_list = [red, c1, c2]
for mechanism in ["VH", "VT", "VHT"]:        
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_axes([0.15,0.55,0.7,0.4])
    ax2 = fig.add_axes([0.15,0.1, 0.7,0.4], sharex = ax1)
    ax1.set_ylabel("log$_{10} |j|$ [mA/cm$^{2}$]")
    ax1.set_ylim(-7,7)
    ax1.tick_params(labelbottom=False)
    ax1.axvline(0, c = "gray", ls = ":")
    ax2.axvline(0, c = "gray", ls = ":")
    ax2.set_ylabel("Tafel slope [mV/dec]")
    ax2.set_xlabel("$E$ [V vs. RHE]")
    ax2.set_ylim(0,150)
    ax2.set_yticks([0,40,80,120])
    ax2.axhline(30, c = "gray", ls = ":")
    ax2.axhline(40, c = "gray", ls = ":")
    ax2.axhline(60, c = "gray", ls = ":")
    ax2.axhline(120, c = "gray", ls = ":")
    ax1.text(0.05,0.9,"A", transform=ax1.transAxes, fontsize = 20, weight = "bold", color = "k")
    ax2.text(0.05,0.9,"B", transform=ax2.transAxes, fontsize = 20, weight = "bold", color = "k")
    for i, par in enumerate(par_list):
        j, logj, E_TS, TS = get_j_HER(**par, mechanism = mechanism)
        ax1.plot(E_lin_HER,logj, c_list[i])
        ax2.plot(E_TS, TS, c_list[i])
    plt.savefig(f"{fig_dir}BE_Comparison_{mechanism}.png")
