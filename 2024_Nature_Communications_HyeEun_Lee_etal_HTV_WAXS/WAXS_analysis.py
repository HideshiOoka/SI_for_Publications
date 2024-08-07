#%% -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:25:47 2022

@author: Hideshi_Ooka
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
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
    
def double_Gaussian(x,A,mu,sigma, baseline):
    y = np.zeros(len(x))
    for i in range(-3,3): # this will consider peaks in the range of -540 to 540 which should definitely be wide enough
        y +=np.exp(-(x-mu-i*180)**2/2/sigma**2)
    y *=A
    y += baseline
    return y

def flat_baseline(x, baseline):
    y = np.zeros(len(x))
    y += baseline
    return y

def fit_double_Gaussian(x,y, fig_name, export_fig = True, mask_threshold = 0.1):
    area = fig_name.split("_")[0]
    i_critical = int(len(y)*mask_threshold)
    y_min = sorted(y)[i_critical]    
    idx = np.where(y < y_min) # detector area
    del_x = x[idx]
    del_y = y[idx]
    x = np.delete(x, idx)
    y = np.delete(y, idx)
    baseline = y_min
    y_raw=y.copy()
    y = savgol_filter(y_raw,11,1)
    A = y.std()
    mu = x[np.where(y == sorted(y)[-i_critical])][0] # the biggest place is prone to noise
    sigma = 10
    if mu <0:
        mu += 180
    init_pars = (A,mu,sigma,baseline)
    try:
        (A,mu,sigma,baseline),cov = curve_fit(double_Gaussian, x, y, p0=init_pars) #, bounds=(0, [10., 180., 90, 10]) makes it so slow
        A = np.abs(A)
        sigma = np.abs(sigma)        
        mu = mu%360
    except RuntimeError:
        print(i)
        # export_fig = True # Uncomment this line to export figures for data which didn't fit well
        baseline,cov = curve_fit(flat_baseline, x, y, p0 = baseline)
        baseline = baseline[0] # pars are returned as a list
        A, mu, sigma = 0,0,180
    y_fit = double_Gaussian(x, A,mu,sigma,baseline)
    r2 = np.sum((y-y_fit)**2)
    if export_fig == True:
        fig_dir = f"Fitted_Figures//{area}//"  
        try:
            os.makedirs(fig_dir)
        except FileExistsError:
            pass
        fig = plt.figure(figsize = (6,4))
        ax = fig.add_axes([0.2,0.2,0.7,0.7])
        # ax.scatter(x,y, c = "k", marker = "o", label = "exp")
        # ax.scatter(x,y_raw, c= "r", marker = "+", s =50)
        ax.scatter(x,y, c = "k", s = 1, label = "exp")
        ax.scatter(del_x,del_y, c = "r", s=1, marker = "x", label = "masked")
        ax.plot(x,y_fit, "b", lw = 2, label = "fitted")
        ax.legend(loc = "upper right")
        # title = file.split("\\")[-1].replace(".h5", f"_{i+1}")
        ax.set_title(fig_name)
        ax.set_xlabel(r"$\theta $ [degrees]")
        ax.set_ylabel("Intensity")
        ax.set_xlim(-185,185)
        ax.set_ylim(-1,11)
        plt.savefig(fig_dir + fig_name + ".png", dpi = 50)
        plt.close() # because too many figures will be opened
    S_exp = get_S(x,y)    
    x_theory = np.linspace(0,180, 720)
    y_theory = double_Gaussian(x_theory, A,mu,sigma,baseline)
    S_theory = get_S(x_theory, y_theory)
    return A, mu, sigma,baseline,r2,S_exp,S_theory

def get_S(x,y):
    I = y[x>0]
    x = x[x>0]
    theta = x/180*np.pi
    sin = np.sin(theta)
    cos = np.cos(theta)    
    denominator = np.sum(I*sin)
    numerator = np.sum(I*sin*cos**2)
    cos2mean = numerator/denominator
    S = 1.5*cos2mean -0.5
    return S

#%%#########################################################
### This code block fits the WAXS data (.dat files) ########
### and generates a CSV with the fitted parameters #########
############################################################
input_files = [f"sample-{2*i+1}-{2*(i+1)}.dat" for i in range(20)]
# input_files = ["sample-1-2.dat","sample-3-4.dat"]
area_name = input_files[0].split("-")[0]
save_file = f"output_files/{area_name}_fitted.csv"
out_txt = "name,A,mu,sigma,baseline,r2,S_exp,S_theory,notes"
for input_file in input_files:
    print(input_file)
    df = pd.read_csv(f"dat_files//{input_file}", sep = "\t", index_col = False).dropna()
    # df = df.iloc[:,:-1]
    N = df.shape[1]//2 # number of samples
    for i in range(N):
        theta, intensity = df.iloc[:,i].values,df.iloc[:,i+N].values
        fig_name = f"{area_name}_"+ df.columns[i][-5:]
        A, mu, sigma,baseline,r2,S_exp,S_theory = fit_double_Gaussian(theta,intensity, fig_name, export_fig = False)
        notes = ""
        if r2 >360:
            notes="check"
        out_txt+= f"\n{fig_name},{A},{mu},{sigma},{baseline},{r2},{S_exp},{S_theory},{notes}"
with open(save_file, "w") as f:
    f.write(out_txt)

#%%###############################################################
### This code block reads the CSV file from the block above ######
### and adds X, Y coordinates based on the WAXS scan direction ###
##################################################################
num_h5 = input_file.split("-")[-1].split(".dat")[0]
num_h5 = int(num_h5) # read until the Nth index
im_per_h5 = 100 # number of images in one h5
h5_per_row = 2 # number of h5s in one row

df = pd.read_csv(save_file)
df = df[df.name !=f"{area_name}_001.1"]
df.index = [int(name[-5:]) for name in df.name]
missing = [i for i in np.arange(1,num_h5*im_per_h5+1) if i not in df.index]
missing_arr = np.zeros((len(missing),df.shape[1]))
missing_arr[:,:] = np.nan
missing_df = pd.DataFrame(missing_arr, index=missing, columns = df.columns)
missing_df["name"] = f"{area_name}_00" + missing_df.index.astype(str)
df = pd.concat([df, missing_df], ignore_index=True)
df = df.sort_index()
A,mu,sigma,baseline,r2,S_exp,S_theory = df.values.T[1:-1].astype(float) 
fwhm = 2*sigma*np.sqrt(2*np.log(2)) # this is degrees units and sometimes exceeds +-180degrees
fwhm = np.clip(fwhm, -180, 180) * np.pi/180 # radian units

G2 = (np.sin(fwhm/2))**2/(2**(2/3)-1)
eta = ((1+G2)**(1/3)-G2**(1/3))**(3/2)
df["FWHM"] = fwhm
df["G2"] = G2
df["eta"] = eta
df.set_index("name")

N, M = num_h5//h5_per_row,im_per_h5*h5_per_row
result = eta * A
result = result.reshape(N, M)
par_name = "eta_x_A"

X,Y = np.meshgrid(np.arange(M),np.arange(N))
for i in range(N):
    if i %2 == 0:
        result[i] = result[i][::-1]
        X[i] = X[i][::-1]
result = result.T
X = X.flatten()
Y = Y.flatten()

df["X"] = X
df["Y"] = Y
cols = ['name', 'X', 'Y', 'A', 'mu', 'sigma', 'baseline', 'S_exp', 'S_theory', 'FWHM', 'G2', 'eta','r2', 'notes']
df = df[cols]
cleaned_file = save_file.replace(".csv","_cleaned.csv")
df.to_csv(cleaned_file, index = False)

#%%#############################################
### This block shows the results as an image ###
################################################
"""
The figure outputted from this block is only for roughly visualizing the numerical data, and was not used to generate any of the final figures in the 2024 Nat. Commun. paper by Dr. Hye-Eun Lee et al. The final figures were generated by Dr. Hye-Eun Lee based on the cleaned csv file. This code was left included so that people using our code to analyze their own WAXS/SAXS data can quickly visualize their data, and is hence not of publication quality. 

Note that in Fig. 2e, some locations not corresponding to the wall structure were masked due to their noise. Both masked and unmasked files can be found here:
https://doi.org/10.5281/zenodo.12788138
No mask was used for Supporting Fig 8.
"""


fig = plt.figure(figsize = (8,4))
ax1 = fig.add_axes([0.1,0.1,0.5,0.8])
result_max = np.nanmax(result)
vmin, vmax = 0,result_max*0.08 #25 #0.1,0.7
im = ax1.imshow(result, vmin = vmin, vmax = vmax, cmap = "jet")
#im = ax1.imshow(result, vmin = 0, vmax = result_max*0.07)
cbar = fig.colorbar(im, ax=ax1)
cbar.ax.set_title(f"{par_name}", fontsize = 20)
cbar.ax.tick_params(labelsize=14) 
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xlim(result.shape[1],0)
ax2 = fig.add_axes([0.7,0.1,0.2,0.8])
ax2.hist(result.flatten(), color="r", bins = 100, alpha = 0.7)
ax2.set_xlim(0,result_max*0.2)
plt.savefig(save_file.replace("fitted.csv",f"{par_name}.png"))
plt.show()
