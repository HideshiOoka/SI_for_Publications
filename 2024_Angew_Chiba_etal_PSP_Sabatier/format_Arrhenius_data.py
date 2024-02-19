#%%
"""
Convert reaction rates (assumed to be kcat at some T) to ln k2 (avg, std) and save the csv
"""
import pandas as pd
import numpy as np
ORG_LIST = ["Hsa","Tel","Cau","Mja","Tth","Eco","Mru","Bsu","Pma","Hth"] 
# Hth was determined from MM plots due to large Km

kcat_Hth_avg = np.array([np.nan,np.nan,np.nan,3.822598923, 22.60817607])
kcat_Hth_std = np.array([np.nan,np.nan,np.nan,0.266026692, 1.751683447])
log_kcat_Hth_avg = np.log(kcat_Hth_avg)
log_kcat_Hth_std = np.abs(np.log(kcat_Hth_std))
# Strictly speaking, the std(log(data)) and log(std(data)) are not the same,
# but error bars of Hth should be small enough to not matter.
# See caption of Fig. S7 for details.

def get_avg_std_of_Arrhenius(file = "Raw_Data//Arrhenius_Raw.csv"):
    df = pd.read_csv(file, index_col = 0)
    orgs = df.index.dropna().tolist()
    N = len(orgs)
    T_list = df.columns.astype(int).tolist()
    NUM_REPEATS = df.shape[0]//N
    NUM_T = len(T_list)
    vals = df.values
    vals = vals.reshape([N, NUM_REPEATS, NUM_T])
    lnk2 = np.log(vals)
    lnk2_avg = np.nanmean(lnk2, axis=1)
    lnk2_avg = np.vstack([lnk2_avg, log_kcat_Hth_avg])
    df_avg = pd.DataFrame(lnk2_avg, columns = T_list, index = ORG_LIST)
    df_avg.to_csv("Analyzed//PSP_lnk2_avg.csv")
    lnk2_std = np.nanstd(lnk2, axis=1)
    lnk2_std = np.vstack([lnk2_std, log_kcat_Hth_std])
    df_std = pd.DataFrame(lnk2_std, columns = T_list, index = ORG_LIST)
    df_std.to_csv("Analyzed//PSP_lnk2_std.csv")
if __name__ == "__main__":
    get_avg_std_of_Arrhenius()