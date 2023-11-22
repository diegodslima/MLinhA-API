from rdkit import Chem
import polars as pl
import pandas as pd
import pickle
from tqdm import tqdm

from src.functions.standardScaler import standardScaler
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

dataset = {"name": names, "smiles": smiles_list}
df_mordred = pd.DataFrame(data=dataset)
print('Calculating Mordred Descriptors... (may take several hours)')
df_mordred = pd.concat([df_mordred, getMordredDescriptors(smiles_list, descriptor_list)], axis=1)

df_mordred = pl.from_pandas(df_mordred)
print(df_mordred)

df_mordred.write_parquet('src/dataset/example/best_ligands_mordred.parquet')

print('loading mtb model...')
with open('src/models/ml-models/inhA-Hgb-lr0.05-possion-iter100.pkl', 'rb') as model_file:
    model_mtb = pickle.load(model_file)
    
print('reading dataset...')
df = pl.read_parquet('src/dataset/example/best_ligands_mordred.parquet')

print(df)

X = df.select(descriptor_list).to_numpy()

print('scaling features...')
X_scaled = standardScaler('src/models/scalers/inha-StandardScaler-33.pkl', X)

smiles = df['smiles']
names = df['name']

print('PREDICTION TIME LETS GOOOO')
predictions = []
with tqdm(total=len(X_scaled)) as pbar:
    for i, descriptors in enumerate(X_scaled):
        pred = model_mtb.predict([descriptors])
        predictions.append(pred[0])
        pbar.update(1)

print('done!')

print('saving as parquet file')
df_pred = pl.DataFrame(data={'names': names, 'smiles': smiles, 'inhA_predictions': predictions})
df_pred.write_parquet('src/predictions/best_ligands_mordred_predictions.parquet')