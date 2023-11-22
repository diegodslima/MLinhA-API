import pandas as pd
import polars as pl
import pickle
from tqdm import tqdm
from rdkit import Chem
from src.functions.standardize import standardize
from src.functions.readSmiles import readSmiles
from src.functions.getMordredDescriptors import getMordredDescriptors
from src.functions.standardScaler import standardScaler


class Dataset:
    def __init__(self, smiles_file_path):
        self.smiles_file_path = smiles_file_path

    def print_path(self):
        print("SMILES File Path:", self.smiles_file_path)
        
    def create_dataframe(self):
        molecules = readSmiles(path=self.smiles_file_path, 
                               delimiter=' ', 
                               titleLine=False)
        
        smiles = [Chem.MolToSmiles(mol) for mol in molecules]
        names = [mol.GetProp("_Name") for mol in molecules]
        standard_smiles = standardize(smiles)

        df = pl.DataFrame(data={"names": names, "smiles": standard_smiles})
        self.dataframe = df
        print('dataframe created')
        
    def print_dataframe(self):
        print(self.dataframe)
        
    def calculate_mordred(self):
        df = self.dataframe
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
        self.mordred_dataframe = pl.from_pandas(df_mordred)
        print('Done.')
        
    def print_mordred_dataframe(self):
        print(self.mordred_dataframe)
        
        
    def mlinha_predict(self):
        
        descriptor_list = {'nX', 'ATS7se', 'ATS7pe', 'AATS0i', 'ATSC4dv', 'ATSC8dv', 'ATSC0s', 'ATSC6s','ATSC7s',
                   'ATSC8v', 'ATSC7se', 'ATSC6pe', 'ATSC8are', 'AATSC0dv', 'AATSC0v', 'AATSC1v', 'SpDiam_Dzse',
                   'VR2_Dzse', 'VR2_Dzpe', 'VR2_Dzare', 'VR2_Dzi', 'C2SP2', 'Xpc-5dv', 'SssssC', 'MIC1',
                   'PEOE_VSA3', 'PEOE_VSA10', 'SlogP_VSA5', 'EState_VSA10', 'MID_X', 'MPC10', 'TpiPC10', 'nRot'}

        with open('src/models/ml-models/inhA-Hgb-lr0.05-possion-iter100.pkl', 'rb') as model_file:
            hgb_model = pickle.load(model_file)
            
        df_descritors = self.mordred_dataframe

        X = df_descritors.select(descriptor_list).to_numpy()

        print('scaling features...')
        X_scaled = standardScaler('src/models/scalers/inha-StandardScaler-33.pkl', X)

        smiles = df_descritors['smiles']
        names = df_descritors['name']
        predictions = []
        pbar = tqdm(total=len(X_scaled), desc="Predicting")

        for _ , descriptors in enumerate(X_scaled):
            pred = hgb_model.predict([descriptors])
            predictions.append(pred[0])
            pbar.update(1)

        pbar.close()

        print('Done.')
        df_pred = pl.DataFrame(data={'names': names, 'smiles': smiles, 'inhA_predictions': predictions})
        self.prediction_dataframe = df_pred
        
    def print_inhA_predictions(self):
        print(self.prediction_dataframe)