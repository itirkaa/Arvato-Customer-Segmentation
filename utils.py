import io
import os
import gc

import pandas as pd
import numpy as np
import pickle
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

cat_cols  = ['CAMEO_DEU_2015', 'D19_LETZTER_KAUF_BRANCHE', 'OST_WEST_KZ']
date_cols = 'EINGEFUEGT_AM'
cust_attr = ['CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP']
rep_value = {'X': ['CAMEO_DEUG_2015'], 
          'XX': ['CAMEO_INTL_2015'], 
          -1: ['AGER_TYP', 'ALTERSKATEGORIE_GROB', 'ANREDE_KZ', 'BALLRAUM', 'CAMEO_DEUG_2015', 'EWDICHTE', 'FINANZTYP', 'FINANZ_ANLEGER', 'FINANZ_HAUSBAUER', 'FINANZ_MINIMALIST', 'FINANZ_SPARER', 'FINANZ_UNAUFFAELLIGER', 'FINANZ_VORSORGER', 'GEBAEUDETYP', 'HEALTH_TYP', 'HH_EINKOMMEN_SCORE', 'INNENSTADT', 'KBA05_ALTER1', 'KBA05_ALTER2', 'KBA05_ALTER3', 'KBA05_ALTER4', 'KBA05_ANHANG', 'KBA05_ANTG1', 'KBA05_ANTG2', 'KBA05_ANTG3', 'KBA05_ANTG4', 'KBA05_AUTOQUOT', 'KBA05_BAUMAX', 'KBA05_CCM1', 'KBA05_CCM2', 'KBA05_CCM3', 'KBA05_CCM4', 'KBA05_DIESEL', 'KBA05_FRAU', 'KBA05_GBZ', 'KBA05_HERST1', 'KBA05_HERST2', 'KBA05_HERST3', 'KBA05_HERST4', 'KBA05_HERST5', 'KBA05_HERSTTEMP', 'KBA05_KRSAQUOT', 'KBA05_KRSHERST1', 'KBA05_KRSHERST2', 'KBA05_KRSHERST3', 'KBA05_KRSKLEIN', 'KBA05_KRSOBER', 'KBA05_KRSVAN', 'KBA05_KRSZUL', 'KBA05_KW1', 'KBA05_KW2', 'KBA05_KW3', 'KBA05_MAXAH', 'KBA05_MAXBJ', 'KBA05_MAXHERST', 'KBA05_MAXSEG', 'KBA05_MAXVORB', 'KBA05_MOD1', 'KBA05_MOD2', 'KBA05_MOD3', 'KBA05_MOD4', 'KBA05_MOD8', 'KBA05_MODTEMP', 'KBA05_MOTOR', 'KBA05_MOTRAD', 'KBA05_SEG1', 'KBA05_SEG10', 'KBA05_SEG2', 'KBA05_SEG3', 'KBA05_SEG4', 'KBA05_SEG5', 'KBA05_SEG6', 'KBA05_SEG7', 'KBA05_SEG8', 'KBA05_SEG9', 'KBA05_VORB0', 'KBA05_VORB1', 'KBA05_VORB2', 'KBA05_ZUL1', 'KBA05_ZUL2', 'KBA05_ZUL3', 'KBA05_ZUL4', 'KBA13_ALTERHALTER_30', 'KBA13_ALTERHALTER_45', 'KBA13_ALTERHALTER_60', 'KBA13_ALTERHALTER_61', 'KBA13_AUDI', 'KBA13_AUTOQUOTE', 'KBA13_BJ_1999', 'KBA13_BJ_2000', 'KBA13_BJ_2004', 'KBA13_BJ_2006', 'KBA13_BJ_2008', 'KBA13_BJ_2009', 'KBA13_BMW', 'KBA13_CCM_1000', 'KBA13_CCM_1200', 'KBA13_CCM_1400', 'KBA13_CCM_0_1400', 'KBA13_CCM_1500', 'KBA13_CCM_1600', 'KBA13_CCM_1800', 'KBA13_CCM_2000', 'KBA13_CCM_2500', 'KBA13_CCM_2501', 'KBA13_CCM_3000', 'KBA13_CCM_3001', 'KBA13_FAB_ASIEN', 'KBA13_FAB_SONSTIGE', 'KBA13_FIAT', 'KBA13_FORD', 'KBA13_HALTER_20', 'KBA13_HALTER_25', 'KBA13_HALTER_30', 'KBA13_HALTER_35', 'KBA13_HALTER_40', 'KBA13_HALTER_45', 'KBA13_HALTER_50', 'KBA13_HALTER_55', 'KBA13_HALTER_60', 'KBA13_HALTER_65', 'KBA13_HALTER_66', 'KBA13_HERST_ASIEN', 'KBA13_HERST_AUDI_VW', 'KBA13_HERST_BMW_BENZ', 'KBA13_HERST_EUROPA', 'KBA13_HERST_FORD_OPEL', 'KBA13_HERST_SONST', 'KBA13_KMH_110', 'KBA13_KMH_140', 'KBA13_KMH_180', 'KBA13_KMH_0_140', 'KBA13_KMH_140_210', 'KBA13_KMH_211', 'KBA13_KMH_250', 'KBA13_KMH_251', 'KBA13_KRSAQUOT', 'KBA13_KRSHERST_AUDI_VW', 'KBA13_KRSHERST_BMW_BENZ', 'KBA13_KRSHERST_FORD_OPEL', 'KBA13_KRSSEG_KLEIN', 'KBA13_KRSSEG_OBER', 'KBA13_KRSSEG_VAN', 'KBA13_KRSZUL_NEU', 'KBA13_KW_30', 'KBA13_KW_40', 'KBA13_KW_50', 'KBA13_KW_60', 'KBA13_KW_0_60', 'KBA13_KW_70', 'KBA13_KW_61_120', 'KBA13_KW_80', 'KBA13_KW_90', 'KBA13_KW_110', 'KBA13_KW_120', 'KBA13_KW_121', 'KBA13_MAZDA', 'KBA13_MERCEDES', 'KBA13_MOTOR', 'KBA13_NISSAN', 'KBA13_OPEL', 'KBA13_PEUGEOT', 'KBA13_RENAULT', 'KBA13_SEG_GELAENDEWAGEN', 'KBA13_SEG_GROSSRAUMVANS', 'KBA13_SEG_KLEINST', 'KBA13_SEG_KLEINWAGEN', 'KBA13_SEG_KOMPAKTKLASSE', 'KBA13_SEG_MINIVANS', 'KBA13_SEG_MINIWAGEN', 'KBA13_SEG_MITTELKLASSE', 'KBA13_SEG_OBEREMITTELKLASSE', 'KBA13_SEG_OBERKLASSE', 'KBA13_SEG_SONSTIGE', 'KBA13_SEG_SPORTWAGEN', 'KBA13_SEG_UTILITIES', 'KBA13_SEG_VAN', 'KBA13_SEG_WOHNMOBILE', 'KBA13_SITZE_4', 'KBA13_SITZE_5', 'KBA13_SITZE_6', 'KBA13_TOYOTA', 'KBA13_VORB_0', 'KBA13_VORB_1', 'KBA13_VORB_1_2', 'KBA13_VORB_2', 'KBA13_VORB_3', 'KBA13_VW', 'KKK', 'NATIONALITAET_KZ', 'ORTSGR_KLS9', 'OST_WEST_KZ', 'PLZ8_ANTG1', 'PLZ8_ANTG2', 'PLZ8_ANTG3', 'PLZ8_ANTG4', 'PLZ8_GBZ', 'PLZ8_HHZ', 'PRAEGENDE_JUGENDJAHRE', 'REGIOTYP', 'RELAT_AB', 'SEMIO_DOM', 'SEMIO_ERL', 'SEMIO_FAM', 'SEMIO_KAEM', 'SEMIO_KRIT', 'SEMIO_KULT', 'SEMIO_LUST', 'SEMIO_MAT', 'SEMIO_PFLICHT', 'SEMIO_RAT', 'SEMIO_REL', 'SEMIO_SOZ', 'SEMIO_TRADV', 'SEMIO_VERT', 'SHOPPER_TYP', 'TITEL_KZ', 'VERS_TYP', 'WOHNDAUER_2008', 'WOHNLAGE', 'W_KEIT_KIND_HH', 'ZABEOTYP'], 
          0: ['ALTERSKATEGORIE_GROB', 'ALTER_HH', 'ANREDE_KZ', 'CJT_GESAMTTYP', 'GEBAEUDETYP', 'HH_EINKOMMEN_SCORE', 'KBA05_BAUMAX', 'KBA05_GBZ', 'KKK', 'NATIONALITAET_KZ', 'PRAEGENDE_JUGENDJAHRE', 'REGIOTYP', 'RETOURTYP_BK_S', 'TITEL_KZ', 'WOHNDAUER_2008', 'W_KEIT_KIND_HH'], 
          9: ['KBA05_ALTER1', 'KBA05_ALTER2', 'KBA05_ALTER3', 'KBA05_ALTER4', 'KBA05_ANHANG', 'KBA05_AUTOQUOT', 'KBA05_CCM1', 'KBA05_CCM2', 'KBA05_CCM3', 'KBA05_CCM4', 'KBA05_DIESEL', 'KBA05_FRAU', 'KBA05_HERST1', 'KBA05_HERST2', 'KBA05_HERST3', 'KBA05_HERST4', 'KBA05_HERST5', 'KBA05_HERSTTEMP', 'KBA05_KRSAQUOT', 'KBA05_KRSHERST1', 'KBA05_KRSHERST2', 'KBA05_KRSHERST3', 'KBA05_KRSKLEIN', 'KBA05_KRSOBER', 'KBA05_KRSVAN', 'KBA05_KRSZUL', 'KBA05_KW1', 'KBA05_KW2', 'KBA05_KW3', 'KBA05_MAXAH', 'KBA05_MAXBJ', 'KBA05_MAXHERST', 'KBA05_MAXSEG', 'KBA05_MAXVORB', 'KBA05_MOD1', 'KBA05_MOD2', 'KBA05_MOD3', 'KBA05_MOD4', 'KBA05_MOD8', 'KBA05_MODTEMP', 'KBA05_MOTOR', 'KBA05_MOTRAD', 'KBA05_SEG1', 'KBA05_SEG10', 'KBA05_SEG2', 'KBA05_SEG3', 'KBA05_SEG4', 'KBA05_SEG5', 'KBA05_SEG6', 'KBA05_SEG7', 'KBA05_SEG8', 'KBA05_SEG9', 'KBA05_VORB0', 'KBA05_VORB1', 'KBA05_VORB2', 'KBA05_ZUL1', 'KBA05_ZUL2', 'KBA05_ZUL3', 'KBA05_ZUL4', 'RELAT_AB', 'SEMIO_DOM', 'SEMIO_ERL', 'SEMIO_FAM', 'SEMIO_KAEM', 'SEMIO_KRIT', 'SEMIO_KULT', 'SEMIO_LUST', 'SEMIO_MAT', 'SEMIO_PFLICHT', 'SEMIO_RAT', 'SEMIO_REL', 'SEMIO_SOZ', 'SEMIO_TRADV', 'SEMIO_VERT', 'ZABEOTYP']}

REMOVE_COLS = ['AGER_TYP', 'KK_KUNDENTYP', 'KKK', 'TITEL_KZ', 'EXTSEL992', 'ALTER_KIND3', 'ALTER_KIND4', 'KBA05_BAUMAX', 'ALTER_KIND2', 'ALTER_KIND1', 'ALTER_HH', 'REGIOTYP', 'CAMEO_DEU_2015']

prefix = 'arvato-segmentation'
bucket = 'sagemaker-ap-south-1-714138043953'


def read_s3_files(files, s3_client):
    """Function to read pickle files from S3
    :param key: file name
    :param prefix: file directory (Default: 'arvato-segmentation')
    :return: data
    """
    data = []
    
    for file in files:
        key = '{}/{}'.format(prefix, file)
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response['Body'].read()
        
        if file[-3:] == 'csv': # Read csv files
            df = pd.read_csv(io.BytesIO(body), encoding='utf-8', sep=';')
        elif file[-4:] == 'xlsx': # Read excel files
            df = df = pd.read_excel(io.BytesIO(body), encoding='utf-8', header=1).dropna(how='all', axis=1)
        else: # Read pickle files
            df = pickle.loads(body)
        data.append(df)
        
    return data


def downcast_dtypes(df, cat=False):
    """ Function to downcast column data types to reduce memory usage.
    :param df: dataframe to downcast
    :param cat: Boolean
    :return: downcasted dataframe
    """
    # if cat is true, downcast non-numeric dtypes
    if cat:
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


def replace_values(df, values=rep_value):
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


def binning_level_data(df):
    df.loc[df['LP_STATUS_GROB']==2,'LP_STATUS_GROB'] = 1 # low income earners
    df.loc[df['LP_STATUS_GROB'].isin([3, 4, 5]),'LP_STATUS_GROB'] = 2 # average earners
    df.loc[df['LP_STATUS_GROB'].isin([6, 7]),'LP_STATUS_GROB'] = 3 # independents
    df.loc[df['LP_STATUS_GROB'].isin([8, 9]),'LP_STATUS_GROB'] = 4 # houseowners
    df.loc[df['LP_STATUS_GROB']==10,'LP_STATUS_GROB'] = 5 # top earners
    
    df.loc[df['LP_FAMILIE_GROB'].isin([3, 4, 5]),'LP_FAMILIE_GROB'] = 3 # independents
    df.loc[df['LP_FAMILIE_GROB'].isin([6, 7, 8]),'LP_FAMILIE_GROB'] = 4 # houseowners
    df.loc[df['LP_FAMILIE_GROB'].isin([9, 10, 11]),'LP_FAMILIE_GROB'] = 5 # top earnersLP_FAMILIE_GROB
    
    return df


def impute(df):
    """Function to impute missing data with most frequent value
    :param df: dataframe
    :return: imputed dataframe
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_cols = df.select_dtypes(numerics).columns.tolist()

    categoric = ['object', 'category']
    categoric_cols = df.select_dtypes(categoric).columns.tolist()
    
    for cols in [numeric_cols, categoric_cols]:
        print('Imputing {} columns...'.format(cols))
        imputer = SimpleImputer(strategy='most_frequent')
        df[cols] = imputer.fit_transform(df[cols])
    
    return df


def scale(df):
    LNR = df['LNR'].copy()
    df.drop(['LNR'], axis=1, inplace=True)
    
    cols_names = df.columns
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    
    df = pd.DataFrame(df, columns=cols_names)
    df['LNR'] = LNR
    
    return df


def preprocess(df):
    """Preprocess data
    :param df: dataframe
    :return: processed dataframe
    """
    # Downcasting all datatypes
    print("Downcasting dataframe...")
    print("Memory usage before downcasting:\n{}".format(df.info(memory_usage=True)))
    df = downcast_dtypes(df, cat=True)
    print("Memory usage after downcasting:\n{}\n".format(df.info(memory_usage=True)))
    
    # Identifying and replacing values representing unknown or missing data
    print("Identifying and replacing values representing unknown or missing data...")
    df = replace_values(df)
    
    # Extracting year from 'EINGEFUEGT_AM' col
    print("Extracting year from datetime cols...")
    df[date_cols] = pd.DatetimeIndex(df[date_cols]).year
    
    # Converting columns to integer type
    cols = ['CAMEO_INTL_2015', 'CAMEO_DEUG_2015']
    print("Converting {} to integer data type...\n".format(cols))
    for col in cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        
    # Removing columns
    print("Removing {} columns...".format(REMOVE_COLS))
    df = df.drop(REMOVE_COLS, axis=1)
    print("Shape after removing data:{}\n".format(df.shape))
    
    # Binning levels data
    print("Bining columns...\n")
    df = binning_level_data(df)
    
    # Imputing missing data
    print("Imputing missing data...")
    df = impute(df)
    
    # Encoding categorical data
    print("Encoding categorical data...")
    print("Shape before encoding", df.shape)
    df = pd.get_dummies(df)
    print("Shape after encoding", df.shape)
    
    # Scaling data
    print("Scaling data...")
    df = scale(df)
    
    # Downcasting all numeric data
    print("Downcasting data...")
    df = downcast_dtypes(df)
    print("Memory usage after downcasting:\n{}\n".format(df.info(memory_usage=True)))

    return df


def predict_clusters(df):
    """Function to reduce dimension of data using pca
    and perform clustering using kmean.
    :param df: dataframe
    :return: dataframe with LNR
    """
    LNR = df['LNR'].copy()
    df.drop(['LNR'], axis=1, inplace=True)
    
    pca = joblib.load('models/pca.save')
    kmeans = joblib.load('models/kmeans.save')
    
    print("Performing Dimensionality Reduction using PCA...")
    print("Shape before PCA:", df.shape)
    df = pca.transform(df)
    df = pd.DataFrame(df)
    print("Shape before PCA:", df.shape)

    print("Finding clusters in data...")
    df_clusters = kmeans.predict(df)
    
    df['CLUSTER'] = df_clusters
    df['LNR'] = LNR
    
    return df