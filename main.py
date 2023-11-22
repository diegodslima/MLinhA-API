from rdkit import Chem
import polars as pl
import pandas as pd

from src.functions.getMordredDescriptors import getMordredDescriptors
from src.functions.standardize import standardize
from src.functions.readSmiles import readSmiles

# READ SMILES (smiles to dataframe)
print('reading smiles file...')
molecules = readSmiles(path='src/dataset/example/best_ligands.smi', delimiter=' ', titleLine=False)

print('getting names and smiles list')
smiles = [Chem.MolToSmiles(mol) for mol in molecules]
names = [mol.GetProp("_Name") for mol in molecules]

print('executing standardize...')
standard_smiles = standardize(smiles)

print('creating dataframe...')
df = pl.DataFrame(data={"names": names, "smiles": standard_smiles})
df.write_parquet(f'src/dataset/example/best_ligands.parquet')
print(df)

# CALCULATE MORDRED DESCRIPTORS (dataframe to mordred)
df = pl.read_parquet('src/dataset/example/best_ligands.parquet')

smiles_list = list(df['smiles'])
names = list(df['names'])

descriptor_list = {'nX', 'ATS7se', 'ATS7pe', 'AATS0i', 'ATSC4dv', 'ATSC8dv', 'ATSC0s', 'ATSC6s','ATSC7s',
                   'ATSC8v', 'ATSC7se', 'ATSC6pe', 'ATSC8are', 'AATSC0dv', 'AATSC0v', 'AATSC1v', 'SpDiam_Dzse',
                   'VR2_Dzse', 'VR2_Dzpe', 'VR2_Dzare', 'VR2_Dzi', 'C2SP2', 'Xpc-5dv', 'SssssC', 'MIC1',
                   'PEOE_VSA3', 'PEOE_VSA10', 'SlogP_VSA5', 'EState_VSA10', 'MID_X', 'MPC10', 'TpiPC10', 'nRot'}

# getMordredDescriptors
dataset = {"name": names, "smiles": smiles_list}
df_mordred = pd.DataFrame(data=dataset)
print('Calculating Mordred Descriptors... (may take several hours)')
df_mordred = pd.concat([df_mordred, getMordredDescriptors(smiles_list, descriptor_list)], axis=1)

df_mordred = pl.from_pandas(df_mordred)
print(df_mordred)

df_mordred.write_parquet('src/dataset/example/best_ligands_mordred.parquet')