#%%
"""
This code converts raw absorption values to enzymatic activity (v) and saves the csv.
It looks at data in the "Abs_Data" directory and saves the new csvs (v) in the "Raw_Data" directory.
It then coverts the v of each individual experiment into avg and std data and saves it into the "Analyzed" directory.
"""
import glob
import pandas as pd
import numpy as np
deltaPi = 44.17 # what is the unit?
ORG_LIST = pd.read_csv("PSP_types.csv", index_col = 0).index.tolist()
N = len(ORG_LIST)

def save_v_and_logv_from_Abs(dir):
    file_list = glob.glob(dir)
    print(file_list)
    for file in file_list:
        df = pd.read_csv(file)
        S, t, E, MW, baseline = df.iloc[:,:5].values.T
        t = t[0]
        E = E
        MW = MW[0]
        Abs = df.iloc[:,5:].values.T
        v = (Abs-baseline)*deltaPi/t/E*MW/60
        v_df = pd.DataFrame(v, columns = S)
        v_file_name = file.replace("Abs","Raw").replace("Raw.csv","v.csv")
        v_df.to_csv(v_file_name)
        logv = np.log10(v_df.values)
        logv_df = pd.DataFrame(logv, columns = S)
        logv_file_name = file.replace("Abs","Raw").replace("Raw.csv","logv.csv")
        # this changes the save dir from Abs_Data to Raw_Data
        logv_df.to_csv(logv_file_name)

def save_v_logv_avg_and_std(T,v_or_logv):
    mean_arr = np.zeros((N,9))
    std_arr = np.zeros((N,9))
    for i, org in enumerate(ORG_LIST):
        try:
            df = pd.read_csv(f"Raw_Data/{T}deg/PSP_{org}_{v_or_logv}.csv", index_col =0)
        except FileNotFoundError:
            print(f"No {v_or_logv} file found for {org} at {T}deg")
            continue
        mean = df.mean(axis = 0)
        std = df.std(axis = 0)
        mean_arr[i] = mean.values
        std_arr[i] = std.values
    S = df.columns.tolist()
    mean_df = pd.DataFrame(mean_arr, columns = S, index = ORG_LIST)    
    std_df = pd.DataFrame(std_arr, columns = S, index = ORG_LIST)    
    mean_df.to_csv(f"Analyzed/PSP_{T}deg_{v_or_logv}_avg.csv")
    std_df.to_csv(f"Analyzed/PSP_{T}deg_{v_or_logv}_std.csv")

if __name__ == "__main__":
    for T in [40,70]:
        save_v_and_logv_from_Abs(fr"Abs_Data/{T}deg/*")
        save_v_logv_avg_and_std(T,"v")
        save_v_logv_avg_and_std(T,"logv")