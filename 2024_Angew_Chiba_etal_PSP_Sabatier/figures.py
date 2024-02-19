#%%
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt, ticker as mticker
from kinetic_analysis import MM, get_r2, get_y_pred

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
R = 8.314 # J/mol/K
fig_dir = "../Figures/"
dblue, blue, gblue, lblue, red, pink, orange =["#033453","#0C7185","#4C837A","#0DAFE1","#910C07","#FF5657","#F48153"]
c_list_9 = ["b","steelblue","g","limegreen", "silver", "grey","orange","tomato","crimson","k"]
PSP_types = pd.read_csv("PSP_types.csv", index_col = 0)
v_40 = pd.read_csv("Analyzed/PSP_40deg_v_avg.csv", index_col = 0)
v_40_std = pd.read_csv("Analyzed/PSP_40deg_v_std.csv", index_col = 0)
PSP_40_pars = pd.read_csv("Analyzed/PSP_40deg_pars_0.5.csv", index_col = 0) # use alpha = 0.5 as the default parameters
logv_40 = pd.read_csv("Analyzed/PSP_40deg_logv_avg.csv", index_col = 0).values
logv_40_std = pd.read_csv("Analyzed/PSP_40deg_logv_std.csv", index_col = 0).values
PSP_Arrhenius_pars = pd.read_csv("Analyzed/PSP_Arrhenius_pars.csv",index_col = 0)
cellulase_pars = pd.read_csv("Analyzed/Cellulases_pars_0.7.csv", index_col = 0)
#%%###################################
### Fig 1 Representative MM Plots ####
######################################
def linspace(S,min_offset=0,max_offset=0):
    return np.linspace(S.min()+min_offset,S.max()+max_offset,101)
indices = np.array([1,8,9])
orgs = PSP_types.index.to_numpy()[indices]
label_list = PSP_types["protein_name"][indices]
c_list = [lblue, orange, dblue]
marker_list = ["D","s","o"]
offset = [-0.2,-0.5,-0.7]
fig = plt.figure(figsize = (8,6))
ax = fig.add_axes([0.2,0.2,0.6,0.7])
for i,org in enumerate(orgs):
    j = indices[i]
    v = v_40.iloc[j]
    std = v_40_std.iloc[j]
    S = v_40.columns.astype(float)
    Km,k2 = PSP_40_pars.iloc[j,:2]
    ax.plot(linspace(S), MM(linspace(S),Km,k2), c_list[i], ls = "--")
    ax.errorbar(S,v, yerr =std, color = c_list[i], capsize=4, fmt=marker_list[i], markersize=7, ecolor='black')
    ax.text(9.5,v[-1]+offset[i],label_list[i],fontsize = 14, color = c_list[i], ha = "right")
ax.set_xlabel("L-P-Ser (mM)")
ax.set_ylabel("$v_0/$[E] (1/s)")
plt.savefig(f"{fig_dir}Fig_1_MM_Examples.png",dpi = 600)

#%%########################################
### Fig 2 Representative Volcano Plots ####
###########################################
def make_volcano(S,logv,Km, logv_std, fig_name,c_list=c_list_9, ls="o", annotate = True): # Fig. 2
    fig = plt.figure(figsize = (8,4))
    ax = fig.add_axes([0.3,0.2,0.4,0.7])
    for j,s in enumerate(S): # change plot color for each S
        color = c_list[j]
        if logv_std.all() != 0:
            ax.errorbar(Km, logv[:,j], yerr =logv_std[:,j], color = color, capsize=4, markersize=7, ecolor='black')
        else:
            ax.plot(Km, logv[:,j], ls, color = color, ms = 5)
    x = Km[np.isfinite(Km)]
    y = logv[np.isfinite(logv)]
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    ax.set_xlim(xmin*10**(-0.5),xmax*10)
    ax.set_ylim(ymin-1.5,ymax+1.3)
    ax.tick_params(axis='x', pad=10)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax.set_xlabel(r"$K_\mathrm{m}$ (mM)")
    ax.set_ylabel("log$_{10} v_0/$[E] (1/s)")
    if annotate == True:
        xq = xmax*2
        logv_q = logv[np.isfinite(logv)]
        yq1 = np.nanmin(logv_q)
        yq2 = np.nanmax(logv_q)
        yq1= np.nanmin(logv_q)
        yq2= np.nanmax(logv_q)
        ax.quiver(xq,yq1,0,yq2-yq1,angles='xy',scale_units='xy',scale=1)
        ax.text(xq/1.2,yq1-0.2, f"{S.min()} mM", ha = "center", va = "top", fontsize = 14)
        ax.text(xq/1.2,yq2+0.3, f"{S.max()} mM", ha = "center", va = "top", fontsize = 14)
    if fig_name =="":
        return fig, ax
    else:
        plt.savefig(f"{fig_dir}{fig_name}",dpi = 600)

S = v_40.columns.astype(float).to_numpy()
Km = PSP_40_pars["Km"].values

make_volcano(S,logv_40, Km, logv_40_std, "Fig_2_Raw_Volcano.png")
extract_indexes = [0,8]
S_extract = S[extract_indexes]
logv_40_extract = logv_40[:, extract_indexes]
logv_40_std_extract  = logv_40_std[:, extract_indexes]
make_volcano(S_extract,logv_40_extract, Km, logv_40_std_extract, "Fig_2_Extracted_Volcano.png", c_list = [orange, blue], ls = "o--")



#%%########################################
### Make Pars Table #######################
###########################################
def extract_C(df, keep_Km_kcat = False):
    alpha = df["alpha"][0]
    df = df[["Km","kcat","C"]]
    if keep_Km_kcat == False:
        df = df[["C"]]
    df = df.rename(columns={"C":alpha})
    return df

def make_pars_table(enzyme):
    """
    This function generates a table of C values with respect to each alpha (0.11, 0.2, 0.5, 0.8) assumed
    There are two columns outputted for each alpha value.
    The first is numerically accurate, the second is a rounded value at 2 decimal points.
    """
    file_list = os.listdir("Analyzed/")
    file_list = [f for f in file_list if enzyme in f and "pars" in f]
    df_list = []
    for i,file in enumerate(file_list):
        df = pd.read_csv("Analyzed/" + file,index_col = 0)
        if i == 0:
            df_list += [extract_C(df, keep_Km_kcat = True)]
        else:
            df_list += [extract_C(df)]
    table_all_pars = pd.concat(df_list, axis = 1)
    table_all_pars.to_csv(f"Analyzed/{enzyme}_Pars_Table.csv")
    table_all_pars.round(2).to_csv(f"Analyzed/{enzyme}_Pars_Table_Rounded.csv")
make_pars_table("PSP_40deg")

#%%########################################
### Fig 3 Plot LFER #######################
###########################################

def plot_LFER(alpha, y_pos_1 = 1, y_pos_2 = 10):
    Km_theory = np.logspace(-2,3)
    PSP_df = pd.read_csv(f"Analyzed/PSP_40deg_pars_{alpha}.csv",index_col = 0)
    dfs = [cellulase_pars,PSP_df]
    c_list = ["k","crimson"]
    marker_list = ["^","x"]
    fig = plt.figure(figsize = (8,6))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    for i in range(2):
        df = dfs[i]
        Km,kcat,alpha,logC,logC_fit, C, C_fit = df.values.T
        logC_fit = logC_fit[0]
        C_fit = C_fit[0]
        alpha = alpha[0]
        ax.plot(Km_theory, C_fit * Km_theory **alpha, c_list[i], ls = "-")
        kcat_pred = C_fit * Km **alpha
        ax.scatter(Km, kcat, c =c_list[i], marker = marker_list[i])
        for j in range(len(Km)):
            ax.plot([Km[j],Km[j]], [kcat[j], kcat_pred[j]],c = c_list[i], ls = ":")
        log_kcat = np.log(kcat)
        log_kcat_avg = np.mean(log_kcat)
        log_kcat_pred = np.log(kcat_pred)

        r2=get_r2(log_kcat_pred,log_kcat)
        print(r2)
        # print(alpha, C_fit, r2, np.std(logC))
        if i == 0:
            ax.text(10,0.05, "Cellulase", c = c_list[i])
            ax.text(0.02,0.05,r"$k_\mathrm{cat}= 0.041 \times K_\mathrm{m}^{0.73}$", c = c_list[i], fontsize = 14)
        else:
            ax.text(0.02,y_pos_1, "PSP", c = c_list[i])
            if alpha <= 0.2:
                ax.text(15,y_pos_2,f"$k_{{cat}}= {{{C_fit:.2f}}} \\times K_\mathrm{{m}}^{{{alpha:.2f}}}$", c = c_list[i], fontsize = 14)
            else:
                ax.text(15,y_pos_2,f"$k_{{cat}}= {{{C_fit:.2f}}} \\times K_\mathrm{{m}}^{{{alpha:.1f}}}$", c = c_list[i], fontsize = 14)                
            # ax.text(0.12,25,"log $C$", c = c_list[i], fontsize =14)
    ax.set_xlim(0.01,500)
    ax.set_ylim(0.001,1000)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$K_\mathrm{m}$ (mM)")
    ax.set_ylabel(r"$k_\mathrm{cat}$ (1/s)")
    ax.tick_params(axis='x', pad=10)
    plt.savefig(f"{fig_dir}Fig_3_LFER_PSP_{alpha}.png", dpi = 600)
plot_LFER(0.1, y_pos_1 = 1, y_pos_2 = 10)
plot_LFER(0.2, y_pos_1 = 2, y_pos_2 = 10)
plot_LFER(0.5, y_pos_1 = 1, y_pos_2 = 20)
plot_LFER(0.8, y_pos_1 = 0.5, y_pos_2 = 80)

#%% Fig 4 Normalized Volcano Plot
def make_normalized_volcano(alpha, enzyme = "PSP_40deg", Km_theory_min = -3,Km_theory_max = 1, save_fig = True):
    Km_theory = np.logspace(Km_theory_min,Km_theory_max,101)
    par_file = f"Analyzed/{enzyme}_pars_{alpha}.csv"
    df = pd.read_csv(par_file, index_col=0)
    Km,kcat,alpha,logC,logC_fit, C, C_fit = df.values.T # to prevent overwriting alpha
    alpha = alpha[0]
    v_file = f"Analyzed/{enzyme}_v_avg.csv"
    v = pd.read_csv(v_file, index_col=0)
    S = v.columns.astype(float).to_numpy()
    v = v.values
    norm_v = (v.T/C).T
    log_norm_v = np.log10(norm_v)
    fig, ax = make_volcano(S,log_norm_v,Km, np.array([0]), "")

    
    for i in range(len(S)):
        v_theory = Km_theory**alpha*S[i]/(Km_theory+S[i])
        log10v_theory = np.log10(v_theory)
        ax.plot(Km_theory, log10v_theory, "--", c = c_list_9[i], zorder = 0)
    ax.set_ylim(-3,1)
    if enzyme == "Cellulases":
        ax.set_ylim(-1.5,2)
    ax.set_ylabel("log$_{10} v_0/$[E]/C (1/s)")
    if save_fig== True:
        plt.savefig(f"{fig_dir}Fig_4_Norm_Volcano_{enzyme}_{alpha}.png", dpi = 600)
    else:
        return fig, ax

make_normalized_volcano(0.1)
make_normalized_volcano(0.2)
make_normalized_volcano(0.5)
make_normalized_volcano(0.8)
make_normalized_volcano(0.7, "Cellulases", Km_theory_max=3)

#%%###############################
### Full MM Plots ################
##################################
def format(s):
    a,b = s.split(" ")
    a = a[0].upper()
    formatted = f"${a}$. ${b}$"
    return formatted

def make_header_list(PSP_types):    
    header_list = []
    orgs = PSP_types.full_org_name.tolist()
    formatted_orgs = [format(org) for org in orgs]
    proteins = PSP_types.protein_name.tolist()
    header_list = [formatted_org+ " "+protein for formatted_org,protein in zip(formatted_orgs,proteins)]
    return header_list

headers = make_header_list(PSP_types)
v = pd.read_csv(f"Analyzed/PSP_40deg_v_avg.csv",index_col = 0)
S = v.columns.values.astype(float)
c_list_full_MM = ["steelblue","tomato"]
fig, axes = plt.subplots(4,3, sharex = "all", tight_layout=True, figsize = (8,8))
axes = np.array(axes)
for t,T in enumerate(["40","70"]):
    pars = pd.read_csv(f"Analyzed/PSP_{T}deg_pars_0.5.csv",index_col = 0).values
    v = pd.read_csv(f"Analyzed/PSP_{T}deg_v_avg.csv",index_col = 0).values
    v_std = pd.read_csv(f"Analyzed/PSP_{T}deg_v_std.csv",index_col = 0).values
    Km,k2 = pars[:,:2].T
    for i,ax in enumerate(axes.reshape(-1)):    
        if i == 9 or i == 11:
            ax.set_axis_off()
            continue
        if i == 10:
            i -=1
        if Km[i] >0: # i.e., if there is relevant data for that temperature and organism
            ax.errorbar(S,v[i], yerr =v_std[i], capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "k", color=c_list_full_MM[t])
            v_theory = MM(linspace(S),Km[i],k2[i])
            ax.plot(linspace(S),v_theory, c_list_full_MM[t])
            y_max = np.nanmax(k2[i])*1.5
            ax.set_ylim(-y_max*0.1, y_max)
        if t == 1:
            ax.set_title(headers[i], fontsize = 10)
            ax.set_xlim(-0.5,11)
            ax.set_xticks(np.arange(0,11,5))
        if i == 9:
            ax.set_xlabel("L-P-Ser (mM)")
        if i == 6:
            ax.set_ylabel("$v_0/$[E] (1/s)")
        ax.tick_params(axis="x",which='major', length=5)
        ax.tick_params(axis="y",which='major', length=5)
plt.savefig(f"{fig_dir}Fig_S2_All_MM_Plots.png", dpi = 300)
#%%###############################
### Km, kcat vs PSP Types ########
##################################
c_list = [lblue, orange, dblue]
df = pd.concat([PSP_types, PSP_40_pars], axis = 1)

y_names = ["Km","kcat"]
y_labels = [r"$K_\mathrm{m}$ (mM)",r"$k_\mathrm{cat}$ (1/s)"]
T_opt = df["T_opt"].values


for i in range(2):
    y_name = y_names[i]
    y = df[y_name].values
    fig = plt.figure(figsize = (6,4))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.plot(T_opt, y, "o", c = "crimson")
    ax.set_xlabel(r"Optimum Growth Temperature ($\degree$C)")
    ax.set_xlim([30,90])
    ax.set_ylabel(y_labels[i])
    plt.savefig(f"{fig_dir}Fig_S3_S4_T_vs_{y_name}.png",dpi = 600)    


fig = plt.figure(figsize = (8,6))
ax = fig.add_axes([0.2,0.2,0.7,0.7])
for i in range(3):
    type_df = df[df["type"] == i+1]
    Km = type_df["Km"].values
    kcat = type_df["kcat"].values
    ax.plot(Km,kcat, "o", c =  c_list[i], label = f"Type {i+1} PSP")
ax.legend(frameon = False, fontsize = 14)
ax.set_xlim(0.05,20)
ax.set_ylim(1,100)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$K_\mathrm{m}$ (mM)")
ax.set_ylabel(r"$k_\mathrm{cat}$ (1/s)")
ax.tick_params(axis='x', pad=10)
ax.legend()
plt.savefig(f"{fig_dir}Fig_S5_Km_kcat_vs_Type.png", dpi = 600)


#%%###############################
### Arrhenius Plot ###############
##################################
types,proteins,full_org_name, T_opt = PSP_types.values.T
lnA2, A2, Ea2 = PSP_Arrhenius_pars.values.T
lnk2 = pd.read_csv("Analyzed/PSP_lnk2_avg.csv", index_col = 0).astype(float)
lnk2_std = pd.read_csv("Analyzed/PSP_lnk2_std.csv", index_col = 0)
T_arr = lnk2.columns.astype(float).to_numpy()
lnk2 = lnk2.values
lnk2_std = lnk2_std.values

T_Kelvin = T_arr + 273.15
h = 0.25
vsp = 0.02
c_list = ["k",orange, blue,red]
x = 1/T_Kelvin
fig = plt.figure(figsize = (8,10))
for i in range(3):
    ax = fig.add_axes([0.2,0.15+(h+vsp)*(2-i),0.7,h])
    indices = np.where(types == i+1)[0]
    for c,j in enumerate(indices):
        y = lnk2[j]
        yerr = lnk2_std[j]
        ax.errorbar(x, y, yerr = yerr, fmt= "o", color = c_list[c], label = proteins[j], ms = 5, ecolor="k", capsize=5)
        y_fit = lnA2[j] - Ea2[j]/R/T_Kelvin
        ax.plot(x,y_fit, "--", c = c_list[c])    
    if i == 2:
        ax.set_xlabel("1/$T$ (K)")
    else:
        ax.tick_params(labelbottom=False)    
    if i == 1:
        ax.set_ylabel(r"log $k_\mathrm{cat}$ (1/s)")
    ax.legend(frameon = False, title = f"Type {i+1} PSP", title_fontsize = 14)
    ax.set_ylim(-3,10)
    ax.text(0.04,0.83,"ABC"[i], transform=ax.transAxes, fontsize = 20, weight = "bold", color = "k")
plt.savefig(f"{fig_dir}Fig_S7_Arrhenius.png")




#%%#####################
###  Make TOC Figure ###
########################
fs = 12
i = 4
c1,c2,c3 = dblue, red,orange
S = v_40.columns.astype(float).to_numpy()[i]
S = np.array([S])
C = PSP_40_pars["C"].values
C_fit = PSP_40_pars["C_fit"].values[0]
Km = PSP_40_pars["Km"].values
# correction = np.log10(C)
v = v_40.iloc[:,i].values
log_v = np.log10(v)
norm_v = (v.T/C)
log_norm_v = np.log10(norm_v)

Km_theory = np.logspace(-3,3,101)
fig = plt.figure(figsize = (5.5,5))
ax = fig.add_axes([0.2,0.2,0.7,0.7])
ax.plot(Km, log_v, "o", color = c1, ms = 5)
log_v_theory = np.log10(MM(S,Km_theory, C_fit*Km_theory**0.5))
ax.plot(Km_theory, log_v_theory, c2)

# draw deviation arrows
log_v_theory = np.log10(MM(S,Km, C_fit*Km**0.5)) # change vector length
for j,x in enumerate(Km):
    y = log_v[j]
    y_theory = log_v_theory[j]
    if np.abs(y-y_theory) > 0.2:
        if y < y_theory:
            y += 0.05
            y_theory -= 0.05
        else:
            y -= 0.05
            y_theory += 0.05
        ax.quiver(x,y,0,y_theory - y, angles='xy', scale_units='xy', scale=1, ls = "--", color = c3, zorder = 0, headwidth = 5, width = 0.005)
        ax.quiver(x,y_theory,0,y-y_theory, angles='xy', scale_units='xy', scale=1, ls = "--", color = c3, zorder = 0, headwidth = 5, width = 0.005)

ax.text(.21,1.7, "Experimental Activity", c = c1, fontsize = fs)
ax.text(0.002, -0.5, "Theoretical\nSabatier Volcano", c = c2, fontsize = fs)
ax.text(2, 1, "Factors which are\nIndependent of\nBinding Affinity", c = c3, fontsize = fs)
ax.set_xlim(0.001,1000)
ax.set_ylim(-1, 2)
ax.tick_params(axis='x', pad=10)
ax.set_xscale("log")
ax.xaxis.set_major_locator(mticker.LogLocator(numticks=3))
ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
ax.set_xlabel(r"$K_\mathrm{m}$ (mM)")
ax.set_ylabel("log$_{10} v_0$/[E] (1/s)")
plt.savefig(f"{fig_dir}TOC.png", dpi = 600)
#%%###############################
### Km vs C ######################
##################################
dfs = [cellulase_pars,PSP_40_pars]
c_list = ["k","crimson"]
marker_list = ["^","x"]

fig = plt.figure(figsize = (8,6))
ax = fig.add_axes([0.2,0.2,0.7,0.7])
for i,df in enumerate(dfs):
    Km = df.Km.values.reshape(-1,1)
    logC = df.logC.values.reshape(-1,1)
    logKm = np.log10(Km)
    ax.plot(logKm, logC, c = c_list[i], marker = marker_list[i], ls = "")
    logC_pred = get_y_pred(logKm, logC)
    r2 = get_r2(logC, logC_pred)
    print(r2)

ax.set_xlabel(r"log$_{10} K_\mathrm{m}$ (mM)")
ax.set_ylabel(r"log$_{10} C$ (1/mM$^{0.5}$/s)")
ax.text(0,-1, "Cellulase", c = c_list[0])
ax.text(0.3,0.9, "PSP", c = c_list[1])
plt.savefig(f"{fig_dir}Fig_S15_Km_vs_C.png",dpi = 600)

#%%########################
###  Make Press Figures ###
###########################
blue = "#0068B7"
orange = "#F39801"
red = "#B10026"
red = "#D91020"
# dblue, blue, gblue, lblue, red, pink, orange =["#033453","#0C7185","#4C837A","#0DAFE1","#910C07","#FF5657","#F48153"]
grey = "#7F7F7F"
fs = 12
i = 4

c1,c2,c3 = blue, red, orange
S = v_40.columns.astype(float).to_numpy()[i]
S = np.array([S])
C = PSP_40_pars["C"].values
C_fit = PSP_40_pars["C_fit"].values[0]
Km = PSP_40_pars["Km"].values
# correction = np.log10(C)
v = v_40.iloc[:,i].values
log_v = np.log10(v)
norm_v = (v.T/C)
log_norm_v = np.log10(norm_v)

Km_theory = np.logspace(-3,3,101)
fig = plt.figure(figsize = (5.5,5))
ax = fig.add_axes([0.2,0.2,0.7,0.7])
ax.plot(Km, log_v, "s", color = c1, ms = 6)
log_v_theory = np.log10(MM(S,Km_theory, C_fit*Km_theory**0.5))
ax.plot(Km_theory, log_v_theory, c2)

# draw deviation arrows
log_v_theory = np.log10(MM(S,Km, C_fit*Km**0.5)) # change vector length
for j,x in enumerate(Km):
    y = log_v[j]
    y_theory = log_v_theory[j]
    if np.abs(y-y_theory) > 0.2:
        if y < y_theory:
            y += 0.05
            y_theory -= 0.05
        else:
            y -= 0.05
            y_theory += 0.05
        ax.quiver(x,y,0,y_theory - y, angles='xy', scale_units='xy', scale=1, ls = "--", color = c3, zorder = 0, headwidth = 5, width = 0.005)
        ax.quiver(x,y_theory,0,y-y_theory, angles='xy', scale_units='xy', scale=1, ls = "--", color = c3, zorder = 0, headwidth = 5, width = 0.005)
# ax.text(.21,1.7, "Experimental Activity", c = c1, fontsize = fs)
# ax.text(0.002, -0.5, "Theoretical\nSabatier Volcano", c = c2, fontsize = fs)
# ax.text(2, 1, "Factors which are\nIndependent of\nBinding Affinity", c = c3, fontsize = fs)
ax.set_xlim(0.001,1000)
ax.set_ylim(-1, 2)
ax.set_xscale("log")
ax.set_xticks([])
ax.set_yticks([])
# plt.savefig(f"{fig_dir}Press_TOC_Blue_Grey_Orange.png",dpi = 600)
plt.savefig(f"{fig_dir}Press_TOC_Blue_Red_Orange.png",dpi = 600)
#%%
fig, ax = make_volcano(S_extract,logv_40_extract, Km, logv_40_std_extract, "", c_list = [blue, red], ls = "o--", annotate = False)

labels = [item.get_text() for item in ax.get_yticklabels()]
labels = [item.replace("2","100").replace("-2","0.01") for item in labels]
ax.set_ylabel("$v_0$/[E] (1/s)")
ax.set_yticklabels(labels)

plt.savefig("Press_Extracted_Volcano.png", dpi = 600)


# %%
