import pandas as pd

def convert_dtypes(df):
    # Define columns and their respective types
    float_features = ['AATS6m', 'ATSC1dv', 'SssCH2', 'SsssCH', 'SaasN', 'SdO', 'PEOE_VSA1',
                      'SMR_VSA3', 'SlogP_VSA5', 'EState_VSA8', 'VSA_EState2', 'MID_N',
                      'TopoPSA(NO)', 'TopoPSA', 'GGI4', 'SRW07', 'SRW09', 'TSRW10']
    
    int_features = ['nAromAtom', 'nAromBond', 'nBondsA', 'C1SP2', 'n5aRing', 'n5aHRing']

    # Convert float features to float64
    df[float_features] = df[float_features].astype('float64')

    # Convert int features to int64
    df[int_features] = df[int_features].astype('int64')

    return df