import pandas as pd
import pyodbc

def read_xlsb_file(source, name_sheet):
    myDriver = 'Microsoft Excel Driver (*.xls, *.xlsx, *.xlsm, *.xlsb)'
    conn_str = (r'DRIVER={' + myDriver + '};'r'DBQ=%s;'r'ReadOnly=1' %source)

    cnxn = pyodbc.connect(conn_str, autocommit=True)

    sql = "SELECT * FROM [{}]".format(name_sheet + "$")
    data = pd.read_sql(sql,cnxn)
    return data