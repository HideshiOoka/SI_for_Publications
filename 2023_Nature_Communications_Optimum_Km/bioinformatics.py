#%%
import numpy as np
import pandas as pd
from scipy import stats
import itertools

def classify_Km_S_data(threshold_list):
    Classified_Km_S_Data = []
    df = pd.read_csv("Km.csv")
    df["logM"] = np.log10(df["[M]"])
    df["logKm"] =np.log10(df["Km"])
    M_counts = df["M"].value_counts(ascending = False)
    L = len(threshold_list)-1    
    for i in range(L):
        M_list = M_counts[(M_counts>=threshold_list[i]) & (M_counts<threshold_list[i+1])].index.tolist()
        sub_df = df[df["M"].isin(M_list)]
        logKm = sub_df["logKm"].values
        logM = sub_df["logM"].values
        Classified_Km_S_Data.append([logKm, logM])
    return Classified_Km_S_Data

def fit_Gaussian(data):
    mu,s = stats.norm.fit(data)
    pdf = stats.norm.pdf(data, loc=mu, scale=s)
    return mu, s, pdf