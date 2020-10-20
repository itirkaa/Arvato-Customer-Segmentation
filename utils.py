import os
import gc

import pandas as pd
import numpy as np

cat_cols = ['CAMEO_DEU_2015', 'D19_LETZTER_KAUF_BRANCHE', 'OST_WEST_KZ']

def downcast_dtypes(df, cat_cols=cat_cols, date_cols='EINGEFUEGT_AM'):
    # Converting datatype to datatime
    df[date_cols] = pd.to_datetime(df[date_cols])

    # Downcasting to categorical value
    df[cat_cols] = df[cat_cols].astype("category")
    
    # Downcasting int values
    int_cols = list(df.select_dtypes(include=["int32", "int64"]).columns)
    if len(int_cols)>=1:
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    
    # Downcasting float values
    float_cols = list(df.select_dtypes(include=["float32", "float64"]).columns)
    if len(float_cols)>=1:
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast="float")
    
    gc.collect()
    
    return df

def read_data(path='Data'):
    """Funtion to read data from given path, if it exists
    :param path: location of the data files
    :return: list of dataframes
    """
    # Reading csv files and removing any completely empty rows or columns 
    if os.path.exists('Clean_data'):
        print("Reading clean pickle files...")
        azdias = pd.read_pickle("Clean_data/azdias.pkl")
        customers = pd.read_pickle("Clean_data/customers.pkl")
    else:
        print("Reading raw data files")
        # Datatypes of the columns containing numbers and 'X' or 'XX' values
        dtypes={'CAMEO_DEUG_2015': 'object',
                'CAMEO_INTL_2015': 'object'}

        azdias = pd.read_csv('Data/Udacity_AZDIAS_052018.csv', sep=';', dtype=dtypes)
        customers = pd.read_csv('Data/Udacity_CUSTOMERS_052018.csv', sep=';', dtype=dtypes)

        print("Changing datatypes..")
        # Reducing size of data
        azdias = downcast_dtypes(azdias)
        customers = downcast_dtypes(customers)
        
    attributes = pd.read_excel('Data/DIAS Attributes - Values 2017.xlsx', header=1).dropna(how='all', axis=1)
    information = pd.read_excel('Data/DIAS Information Levels - Attributes 2017.xlsx', header=1).dropna(how='all', axis=1)
    
    print("Completed!")
    return azdias, customers, attributes, information



def replace_values(df, values):
    """Function to replace given values in given columns with nans.
    :param df: dataframe to perform operations on
    :param values: dictionary of values(key) to be replaced with 
    nan in the given columns(value)
    
    :return: dataframe with replaced values
    """
    
    for value, cols in values.items():
        to_replace = {value: np.nan}
        df[cols] = df.loc[:, cols].replace(to_replace)
        print("Replaced {} with nan in following column(s):{}".format(value, cols))
        
    return df

