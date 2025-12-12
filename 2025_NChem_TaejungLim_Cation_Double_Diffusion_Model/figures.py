#%%
"""
This code generates the Levich plot (A), Koutecky-Levich plot (B), and voltammograms (C) based on the double diffusion model. Experimental datapoints are shown as scatter plots and the theoretical lines are shown in solid lines. 
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
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
red = "#FA3C3C" # Li
orange = "#F08228" # Na
grey = "#666666" #H
blue = "#1E3CFF" #K
purple = "#6E00DC" #Cs
c_list = [red,orange,grey,blue,purple]
Levich_j = pd.read_csv("simulation_results/Levich_j.csv", index_col = 0)
pars = pd.read_csv("simulation_results/parameters_from_KL.csv", index_col = 0)
sim = pd.read_csv("simulation_results/simulated_3600.csv", index_col = 0)
ion_list = ["Li","Na","H","K","Cs"]
rpm_arr = np.array([1225,1600, 2500, 3600])
radps_arr = rpm_arr*0.10472 # convert rpm to radian per second
sqrt_radps_arr = np.sqrt(radps_arr)


fig = plt.figure(figsize=(12,6))
x,y,w,h,hsp = 0.1, 0.2, 0.2, 0.7,0.14
ax1 = fig.add_axes([x,y,w,h]) 
ax2 = fig.add_axes([x+w+hsp,y,w,h]) 
ax3 = fig.add_axes([x+(w+hsp)*2,y,w,h]) 

for i, ion in enumerate(ion_list):
    a = pars["a"].iloc[i]
    b = pars["b"].iloc[i]
    jlim = Levich_j.iloc[i]

    file = f"raw_data/50mM_{ion}Cl_3600rpm_norm.csv"
    df = pd.read_csv(file)
    E = df.iloc[:,3].values
    jCER = df.iloc[:,6].values
    ax3.plot(E[::50],jCER[::50], "d", c = c_list[i], ms = 4)
    ax3.plot(sim.iloc[:,i], c = c_list[i] )

    x_ax1 = np.linspace(0, 25)
    ax1.plot(x_ax1,1/(a/x_ax1+b), c = c_list[i])
    ax1.plot(sqrt_radps_arr, jlim, "d", c = c_list[i])

    x_ax2 = np.linspace(0, 0.12)
    ax2.plot(x_ax2,a*x_ax2+b, c = c_list[i])
    ax2.plot(1/sqrt_radps_arr, 1/jlim, "d", c = c_list[i])


ax1.set_xlabel(r"$\sqrt{\omega}$ [rad/s]")
ax1.set_ylabel(r"$j_\mathrm{lim}$ [mA/cm$^2$]")
ax1.set_xlim(0,25)
ax1.set_ylim(0,120)
ax1.set_xticks(np.linspace(0,20,3))
ax1.set_yticks(np.linspace(0,120,4))

ax2.set_xlabel(r"$1 / \sqrt{\omega}$ [rad/s]")
ax2.set_ylabel(r"$1 / j$ [mA/cm$^2$]")
ax2.set_xlim(0,0.12)
ax2.set_xticks(np.arange(0,0.1, 0.03))
ax2.set_ylim(0,0.03)
ax2.set_xticks(np.linspace(0,0.1,3))
ax2.set_yticks(np.linspace(0,0.03,4))
ax2.tick_params(axis='x', which='major', pad=8)
ax2.tick_params(axis='y', which='major', pad=8)

ax3.set_xlabel(r"$E-iR$ [V vs. RHE]")
ax3.set_ylabel(r"$j$ [mA/cm$^2$]")
ax3.set_xlim(1.4,1.8)
ax3.set_ylim(-5,100)


ax1.text(0.05,0.9,"A", transform=ax1.transAxes, fontsize = 28, weight = "bold")
ax2.text(0.05,0.9,"B", transform=ax2.transAxes, fontsize = 28, weight = "bold")
ax3.text(0.05,0.9,"C", transform=ax3.transAxes, fontsize = 28, weight = "bold")
plt.savefig("Simulations.png")