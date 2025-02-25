import sqlite3
import modules.alim.parameters.user_parameters as up
import sqlalchemy
global engine

def create_database():
    global engine
    sqlite3.connect(up.database_output_path)
    engine = sqlalchemy.create_engine('sqlite:///' + up.database_output_path)

def insert_df_in_databse(data, name_table):
    data.to_sql(name_table, con=engine, if_exists='fail', index=False)