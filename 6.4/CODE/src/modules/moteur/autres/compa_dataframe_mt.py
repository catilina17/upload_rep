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
import re


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def change_indic(data, col):
    data["temp"] = [int(re.findall("\d+", x)[0]) if hasNumbers(x) else "" for x in data[col]]
    data["temp"] = [str(int(x / 12)) if x!="" else "" for x in data["temp"]]
    data["temp2"] = [''.join([i for i in x if not i.isdigit()]) for x in data[col]]
    data[col] = [x + y for x, y in zip(data["temp2"], data["temp"])]

    data = data.drop(["temp", "temp2"], axis=1)

    return data
 

dir1=r"C:\Users\TAHIRIH\Desktop\Nouveau dossier\Compil_BPCE__PYTHON"

dir2=r"C:\Users\TAHIRIH\Desktop\Nouveau dossier\Compil_BPCE__PYTHON_REF"

files1=["CompilPN.csv"]
files2=["CompilPN.csv"]

for i in range(0,len(files1)):

    f=files1[i]
    
    f2=files2[i]

    list_month=["M0"+str(i) if i<=9 else "M" +str(i) for i in range(0,120)]
    
    df1=pd.read_csv(open(pathlib.Path(dir1, f), 'r'),sep=";",decimal=",",encoding="latin-1")

    excl=["index","SC","DIM NSFR 1","CT_MLT_LIQ","M120"]

    df1=df1.fillna(0).round(0)[[x for x in df1.columns if x not in excl]].copy()

    df1=change_indic(df1,pa.NC_PA_IND03)

    df1=df1.sort_values(by=["CLE","IsIG","IND02",pa.NC_PA_IND03])

    df1=df1.reset_index(drop=True)

    df2=pd.read_csv(open(pathlib.Path(dir2, f2), 'r'),sep=";",decimal=",")
    df2=df2.fillna(0).round(0)[[x for x in df2.columns if x not in excl]].copy()
    #df2[["M0"+str(i) if i<=9 else "M" +str(i) for i in range(0,121)]]=df2[["M0"+str(i) if i<=9 else "M" +str(i) for i in range(0,121)]].astype(np.float64).round(0)

    df2=df2[[x for x in df2.columns if x!='M121' and "Unnamed" not in x]]

    df2=df2.sort_values(by=["CLE","IsIG","IND02",pa.NC_PA_IND03])

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



