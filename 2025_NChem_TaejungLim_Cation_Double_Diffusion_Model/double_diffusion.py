#%%
import glob
import pandas as pd
import numpy as np
##########################################
### Physical Assumptions #################
##########################################
nu = 0.01014031 # cm2/s, 1$ H2SO4
C2 = 0.00005 # 50 mM, mol/cm3  
n = 2
F = 96485000 # mC/mol
nFC2 = n*F*C2

##########################################
### The following parameters for CER #####
### charge transfer were chosen so that ##
### simulations match experiments ########
##########################################
alpha_CER = 0.8
aFRT = alpha_CER*96485/8.314/300
k0 = 20
Eeq = 1.81 # Choosing k0 automatically chooses Eeq (and vice versa)
##########################################
### The cation layer thickness was #######
### chosen as an example and does not ####
### influence simulated CVs ##############
### Only alpha, beta prime influence CV ##
##########################################
L1 = 1E-7 # cm unit, equivalent to 1 nm 
# The thickness of L1 was chosen as an example to facilitate physical interpretation. It has no experimental basis.


##########################################
### Reading the Experimental Data ########
##########################################
all_files = glob.glob("raw_data/*.csv")
all_files = sorted(all_files)
ion_list = ["Li","Na","H","K","Cs"]
target_E_list = [1.67, 1.69, 1.71, 1.73, 1.75]
rpm_arr = np.array([1225,1600, 2500, 3600])
radps_arr = rpm_arr*0.10472 # convert rpm to radian per second
sqrt_radps_arr = np.sqrt(radps_arr)
I = len(ion_list)
R = len(radps_arr)

def get_j(E,j,target_E, anode_only = True):
    if anode_only == True:
        deltaE = E[1:] - E[:-1]
        E = E[:-1][deltaE>0]
    idx = (np.abs(E - target_E)).argmin()
    if np.abs(E[idx] - target_E) < 0.01: # within 10 mV
        return j[idx]
    else:
        return 1

def fit_line(x,y, zero_intercept = False):
    n = len(x)
    X = np.ones((n,2))
    X[:,0] = x
    if zero_intercept == True:
        X = X[:,0]
    XTXinv = np.linalg.inv(X.T@X)
    pars = XTXinv@(X.T@y)
    if zero_intercept == False:
        a,b = pars
    else:
        a = pars
        b = 0
    return a,b

##############################################
### Get Parameters from KL Plot ##############
##############################################
Levich_j_arr = np.zeros((I,R))
a_arr = np.zeros(I)
b_arr = np.zeros(I)
for i, ion in enumerate(ion_list):
    target_E = target_E_list[i]
    for r, radps in enumerate(radps_arr):
        rpm = rpm_arr[r]
        file = f"raw_data\\50mM_{ion}Cl_{rpm}rpm_norm.csv"
        df = pd.read_csv(file)
        E = df.iloc[:,3].values
        jCER = df.iloc[:,6].values
        Levich_j_arr[i,r] = get_j(E,jCER, target_E)
    a,b = fit_line(1/sqrt_radps_arr, 1/Levich_j_arr[i])
    a_arr[i] = a
    b_arr[i] = b
alpha_arr = nFC2*b_arr
betaprime_arr = nFC2*a_arr
##############################################
### Save Parameters to CSV ###################
##############################################
Levich_df = pd.DataFrame(index = ion_list, columns = rpm_arr, data = Levich_j_arr)
Levich_df.to_csv("simulation_results/Levich_j.csv")

par_df = pd.DataFrame(index = ion_list)
par_df["a"] = a_arr
par_df["b"] = b_arr
par_df["alpha"] = alpha_arr
par_df["betaprime"] = betaprime_arr
par_df["beta_3600"] = betaprime_arr/sqrt_radps_arr[-1]
par_df["L1"] = L1
par_df["D1"] = L1/alpha_arr
par_df["D2"] = (1.61*nu**(1/6)/betaprime_arr)**1.5
par_df.to_csv("simulation_results/parameters_from_KL.csv")

##############################################
### Simulate the CV ##########################
##############################################
alpha_arr = par_df["alpha"].values # jK2 = nFC2*k
betaprime_arr = par_df["betaprime"].values
def get_j(alpha,beta,k):
    jK2 = nFC2*k
    C1C2 = (k*alpha+1)/(k*alpha+k*beta+1) #value of C1/C2
    jL1 = nFC2*C1C2/alpha
    jK1 = C1C2*jK2
    j = (1/jL1+1/jK1)**(-1)
    return j

E_arr = np.linspace(1.2,1.75)
k = k0*np.exp(aFRT*(E_arr-Eeq))

for r, radps in enumerate(radps_arr):
    simulation_results = np.zeros((len(E_arr),I))
    rpm = rpm_arr[r]
    beta_arr = betaprime_arr/ np.sqrt(radps)
    for i, ion in enumerate(ion_list):
        alpha = alpha_arr[i]
        beta =beta_arr[i]
        simulation_results[:,i] = get_j(alpha,beta,k)
    df = pd.DataFrame(index = E_arr, columns = ion_list, data = simulation_results)
    df.to_csv(f"simulation_results/simulated_{rpm}.csv")


