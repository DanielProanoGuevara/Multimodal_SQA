# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:22:23 2024

@author: danie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def RMS(x):
    return np.sqrt(np.mean(x**2))


def dummyScoring(x):
    correlation_coeff = np.corrcoef(x['ECG_Lobes'], x['PCG_Lobes'])[0][1]
    overlap = np.where(x['ECG_Lobes'] > 0, x['PCG_Lobes'], 0)
    residues = np.where(x['ECG_Lobes'] > 0, 0, x['PCG_Lobes'])
    
    rms = RMS(overlap)
    area_score = np.trapz(overlap)
    
    residues_area = np.trapz(residues)
    rms_residues = RMS(residues)
    relation_area = area_score/residues_area
    relation_rms = rms/rms_residues
    
    return [correlation_coeff, rms, area_score, rms_residues, residues_area, 
            relation_rms, relation_area] 
    



df_db = pd.read_pickle("./processedDataBase.pkl") 


manualQScores = pd.read_excel('../DatasetCHVNGE/SignalQuality.xlsx', 
                              sheet_name='Total')

manualQScores = manualQScores.sort_values(by=['Trial', 'Spot'])
manualQScores.reset_index(drop=True, inplace=True)

score_Results  = df_db.apply(dummyScoring, axis=1)
score_Results = np.array(score_Results.to_list())
score_labels = ['Correlation_Coefficient', 'Ovelap_RMS', 'Overlap_AUC', 
                'Residues_RMS', 'Residues AUC', 'Relation_RMS', 'Relation_AUC']

scores_df = pd.DataFrame(score_Results, columns = score_labels)

df_db = pd.concat([df_db, scores_df], axis=1)

df_db = pd.concat([df_db, manualQScores['MINMAX_Q']], axis=1)

# Plots
#fig, axes = plt.subplots(nrows=1, ncols=7)

for i, column in enumerate(df_db.columns[6:13]):
    #df_db.boxplot(column, by='MINMAX_Q', ax=axes[i]) #When binary scoring
    df_db.plot.scatter(x='MINMAX_Q', y=column)
    #axes[i].set_title(column)
    plt.tight_layout()
    
# plt.tight_layout()
# plt.show()