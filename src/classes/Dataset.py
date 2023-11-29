import pandas as pd
import polars as pl
import pickle
from rdkit import Chem
from src.functions.removeMissingRows import removeMissingRows
from src.functions.convertDtypes import convertDtypes
from src.functions.splitIntFromFloat import splitIntFromFloat
from src.functions.standardize import standardize
from src.functions.readSmiles import readSmiles
from src.functions.getMordredDescriptors import getMordredDescriptors
from src.functions.rewriteSmilesFile import rewriteSmilesFile


class Dataset:
    def __init__(self, smiles_filename):
        self.smiles_filename = smiles_filename
        self.dataframe = None
        self.mordred_dataframe = None
        self.descriptor_list = None
        self.inha_prediction = None
        
    def create_dataframe(self):
        rewriteSmilesFile(f"./src/temp/{self.smiles_filename}", 
                          f"./src/temp/new-{self.smiles_filename}")
        
        molecules = readSmiles(path=f"./src/temp/new-{self.smiles_filename}", 
                               delimiter=' ', 
                               titleLine=False)
        
        smiles = [Chem.MolToSmiles(mol) for mol in molecules]
        names = [mol.GetProp("_Name") for mol in molecules]
        standard_smiles = standardize(smiles)

        df = pl.DataFrame(data={"names": names, "smiles": standard_smiles})
        self.dataframe = df
        print('Dataframe created.')
        
    def calculate_mordred(self):
        df = self.dataframe
        smiles_list = list(df['smiles'])
        names = list(df['names'])

        self.descriptor_list = {
            'AATS6m', 'ATSC1dv', 'SssCH2', 'SsssCH', 'SaasN', 'SdO', 'PEOE_VSA1',
             'SMR_VSA3', 'SlogP_VSA5', 'EState_VSA8', 'VSA_EState2', 'MID_N',
             'TopoPSA(NO)', 'TopoPSA', 'GGI4', 'SRW07', 'SRW09', 'TSRW10',
             'nAromAtom', 'nAromBond', 'nBondsA', 'C1SP2', 'n5aRing', 'n5aHRing'
            }

        dataset = {"name": names, "smiles": smiles_list}
        df_mordred = pd.DataFrame(data=dataset)
        print('Calculating Mordred Descriptors... (may take several hours)')
        df_mordred = pd.concat([df_mordred, getMordredDescriptors(smiles_list, 
                                                                  self.descriptor_list)], axis=1)

        df_mordred = removeMissingRows(df_mordred)
        
        self.mordred_dataframe = pl.from_pandas(df_mordred)
        print('Mordred descriptors calculated.')
        
    def mlinha_predict(self):
        with open('./src/models/ml-models/mlp_inha_model.pkl', 'rb') as model_file:
            mlp_model = pickle.load(model_file)
            
        df_features = self.mordred_dataframe.to_pandas().iloc[:, 2:]       
        df_features = convertDtypes(df_features)
            
        float_features, int_features = splitIntFromFloat(df_features)
        
        df_float = df_features[float_features]
        df_int = df_features[int_features]
        
        with open('./src/models/scalers/std-scaler-inhA-small-nov23.pkl', 'rb') as model_file:
            std_scaler = pickle.load(model_file)
            
        with open('./src/models/scalers/int-scaler-inhA-small-nov23.pkl', 'rb') as model_file:
            int_scaler = pickle.load(model_file)

        df_float_scaled = pd.DataFrame(data=std_scaler.transform(df_float),
                                       columns=df_float.columns)
        
        df_int_scaled = pd.DataFrame(data=int_scaler.transform(df_int),
                                       columns=df_int.columns)
        
        df_all_features = pd.concat([df_float_scaled, df_int_scaled], axis=1)

        smiles = self.mordred_dataframe['smiles']
        names = self.mordred_dataframe['name']    
        predictions = mlp_model.predict(df_all_features)

        print('Done.')
        df_pred = pl.DataFrame(data={'name': names, 'smiles': smiles, 'inhA_pred_pIC50': predictions})
        self.inha_prediction = df_pred