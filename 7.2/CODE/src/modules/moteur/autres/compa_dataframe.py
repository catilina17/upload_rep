# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:51:44 2020

@author: Hossayne
"""

# -*- coding: utf-8 -*-

"""

Created on Mon Feb 10 11:19:15 2020

 

@author: TAHIRIH

"""


import pandas as pd

import pathlib

import sys

import numpy as np

 

dir2=r"C:\Users\HOSSAYNE\Documents\BPCE_ARCHIVES\RESULTATS\COMPA\MOTEUR_DAR-2022331_EXEC-20220830.2030.23\BP\LOL"

dir1=r"C:\Users\HOSSAYNE\Documents\BPCE_ARCHIVES\RESULTATS\COMPA\MOTEUR_NEW\BP\LOL"

files1=["CompilPN.csv"]
files2=["CompilPN.csv"]

for i in range(0,len(files1)):

    f=files1[i]
    
    f2=files2[i]

    list_month=["M0"+str(i) if i<=9 else "M" +str(i) for i in range(0,241)]
    
    df1=pd.read_csv(open(pathlib.Path(dir1, f), 'r'),sep=";",decimal=",",encoding="latin-1")

    excl=["index","SC"]

    df1=df1.fillna(0).round(0)[[x for x in df1.columns if x not in excl]].copy()

    #if "PN" in files1[0]:
        #df1 = df1[~df1["CONTRAT"].str.contains("AJUST")]
    #df1=df1[~df1[pa.NC_PA_IND03].str.contains("GP LQ EF")]
    #df1=df1[~df1[pa.NC_PA_IND03].str.contains("GP LQ EM")]
        #df1=df1.sort_values(by=["CLE","IsIG","IND02",pa.NC_PA_IND03])

    df1=df1.reset_index(drop=True)

    df2=pd.read_csv(open(pathlib.Path(dir2, f2), 'r'),sep=";",decimal=",")

    df2=df2.fillna(0).round(0)[[x for x in df2.columns if x not in excl]].copy()
    #df2[["M0"+str(i) if i<=9 else "M" +str(i) for i in range(0,121)]]=df2[["M0"+str(i) if i<=9 else "M" +str(i) for i in range(0,121)]].astype(np.float64).round(0)

    #df2=df2[[x for x in df2.columns if x!='M121' and "Unnamed" not in x]]

    df1 = df1[df1[pa.NC_PA_IND03].isin(df2[pa.NC_PA_IND03].values.tolist())]



    df2=df2.sort_values(by=["CLE","IsIG","IND02",pa.NC_PA_IND03])
    df1=df1.sort_values(by=["CLE","IsIG","IND02",pa.NC_PA_IND03])

    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    #if "PN" in files2[0]:
        #df2 = df2[~df2["CONTRAT"].str.contains("AJUST")]
    #df2=df2[~df2[pa.NC_PA_IND03].str.contains("GP LEF EF")]
    #df2=df2[~df2[pa.NC_PA_IND03].str.contains("GP LEF EM")]
        #df2=df2.sort_values(by=["CLE","IsIG","IND02",pa.NC_PA_IND03])

    df2=df2.reset_index(drop=True)
    
    print(df1.columns.tolist())
    print(df2.columns.tolist())

    print(df1.shape[0])
    print(df2.shape[0])
    
    df1.columns=df2.columns
    
    df2=df2.head(df1.shape[0])
    df1=df1.head(df2.shape[0])
    
    print(df1.shape)
    print(df2.shape)
    
    ne_stacked = (df1 != df2).stack()

    changed = ne_stacked[ne_stacked]

    changed.index.names = ['id', 'col']

    difference_locations = np.where(df1 != df2)

    changed_from = df1.values[difference_locations]

    changed_to = df2.values[difference_locations]

    result=pd.DataFrame({'from': changed_from, 'to': changed_to}, index=changed.index)


    result=result.reset_index()
    result_num=result[result["col"].isin(list_month)]
    result_qual=result[~(result["col"].isin(list_month))]
    result_qual=result_qual[result_qual["col"]!="IND02"]

    result_num[["from","to"]]=result_num[["from","to"]].astype(np.float64)

    result_qual.to_csv(dir1+"\diff_var_qual_vba_python" + f.replace(".csv","") + ".csv",sep=";",decimal=",")

    result_num=result_num[abs(result_num["from"]-result_num["to"])>1]

    result_num.to_csv(dir1+"\diff_var_num_vba_python" + f.replace(".csv","") + ".csv",sep=";",decimal=",")

    sys.exit(0)

    result_from=result_num[["id","col","from"]].copy().sort_values(by=["from"]).rename(columns={"id":"id1","col":"col1"}).reset_index(drop=True)
    result_to=result_num[["id","col","to"]].copy().sort_values(by=["to"]).rename(columns={"id":"id2","col":"col2"}).reset_index(drop=True)

    result_num=pd.concat([result_from,result_to],axis=1)

    print(result_num)
    result_num=result_num[abs(result_num["from"]-result_num["to"])>1]

    result_num.to_csv(dir1+"\diff_var_num_vba_python" + f.replace(".csv","") + ".csv",sep=";",decimal=",")



