This is the ReadMe for the 2024 Angew paper by Chiba, Ooka, Nakamura, and others.

Python Code:
- figures.py: All figures were generated using this code. PNG files are saved into the ../Figures directory. Exceptions are the gel photograph of Fig. S1 and the TOC arts which were edited using PPT.
- kinetic_analysis.py: This code is required to run figures.py. It includes the Michaelis-Menten and Arrhenius equations used for fitting and numerical simulations. The code used to calculation of C based on specific values of alpha is also included in this file. Outputs are saved into "Analyzed" folder.
- format_abs_data.py: This converts absorbance data in the Abs_Data directory to avgs and stds which is saved in the "Raw_Data" directory.
- format_Arrhenius.py: This converts the rates in Raw_Data//Arrhenius_Raw.csv into Analyzed//PSP_lnk2_avg.csv and Analyzed//PSP_lnk2_std.csv. The file Arrhenius_Raw.csv is itself raw data (not generated using python).

Due to the automated nature of producing similar figures using python (such as scaling relationship between Km and kcat at different alpha values), the names of several figures were changed manually.

Fig_S3_S4_T_vs_Km.png-->
Fig_S3_T_vs_Km.png

Fig_S3_S4_T_vs_kcat.png-->
Fig_S4_T_vs_kcat.png

Fig_2_Raw_Volcano.png --> 
Fig_S6_Raw_Volcano.png 

Fig_3_LFER_PSP_0.1147903994929156.png-->
Fig_S8_LFER_PSP_0.1147903994929156.png

Fig_4_Norm_Volcano_PSP_40deg_0.1147903994929156.png-->
Fig_S9_Norm_Volcano_PSP_40deg_0.1147903994929156.png

Fig_3_LFER_PSP_0.2.png -->
Fig_S10_LFER_PSP_0.2.png

Fig_4_Norm_Volcano_PSP_40deg_0.2.png-->
Fig_S11_Norm_Volcano_PSP_40deg_0.2.png

Fig_3_LFER_PSP_0.8.png-->
Fig_S12_LFER_PSP_0.8.png

Fig_4_Norm_Volcano_PSP_40deg_0.8.png-->
Fig_S13_Norm_Volcano_PSP_40deg_0.8.png

Fig_4_Norm_Volcano_Cellulases_0.734893249745322.png-->
Fig_S14_Norm_Volcano_Cellulases_0.734893249745322.png