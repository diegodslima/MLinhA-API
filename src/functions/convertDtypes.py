def convertDtypes(df):
    float_features = ['AATS6m', 'ATSC1dv', 'SssCH2', 'SsssCH', 'SaasN', 'SdO', 'PEOE_VSA1',
                      'SMR_VSA3', 'SlogP_VSA5', 'EState_VSA8', 'VSA_EState2', 'MID_N',
                      'TopoPSA(NO)', 'TopoPSA', 'GGI4', 'SRW07', 'SRW09', 'TSRW10']
    
    int_features = ['nAromAtom', 'nAromBond', 'nBondsA', 'C1SP2', 'n5aRing', 'n5aHRing']

    df[float_features] = df[float_features].astype('float64')
    df[int_features] = df[int_features].astype('int64')

    return df