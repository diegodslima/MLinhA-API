import pandas as pd

def removeMissingRows(df):
    int_features = ['nAromAtom', 'nAromBond', 'nBondsA', 'C1SP2', 'n5aRing', 'n5aHRing']
    for column in df.columns:
        if (df[column].dtype == 'object') and (column != 'smiles' and column != 'name'):       
            df[column] = pd.to_numeric(df[column], errors='coerce')
            df = df.dropna(subset=[column])
            if column in int_features:
                df.loc[:, column] = df[column].astype(int)
            else:
                df.loc[:, column] = df[column].astype(float)

    return df