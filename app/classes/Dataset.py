from pathlib import Path
import pandas as pd
import polars as pl
import pickle
from rdkit import Chem
from app.functions.bitStringToArray import bitStringToArray
from app.functions.removeMissingRows import removeMissingRows
from app.functions.convertDtypes import convertDtypes
from app.functions.splitIntFromFloat import splitIntFromFloat
from app.functions.standardize import standardize
from app.functions.readSmiles import readSmiles
from app.functions.getMordredDescriptors import getMordredDescriptors
from app.functions.getMorganFingerprint import getMorganFingerprints
from app.functions.rewriteSmilesFile import rewriteSmilesFile

class Dataset:
    # MLP_MODEL_PATH = 'app/models/ml-models/mlp_inha_model.pkl'
    STD_SCALER_PATH = 'app/models/scalers/std-scaler-inhA-A5-700w-jan24-dropNApchembl.pkl'
    #INT_SCALER_PATH = 'app/models/scalers/int-scaler-inhA-small-nov23.pkl'

    def __init__(self, smiles_filename):
        self.X = None
        self.y = None
        self.smiles_filename = smiles_filename
        self.dataframe = None
        self.mordred_dataframe = None
        self.descriptor_list = None
        self.inha_prediction = None
        self.fingerprints = None

    def create_dataframe(self):       
        current_directory = Path.cwd()
        temp_directory = current_directory / "temp"
        
        rewriteSmilesFile(f"{temp_directory}/{self.smiles_filename}", 
                          f"{temp_directory}/new-{self.smiles_filename}")
        
        molecules = readSmiles(path=f"{temp_directory}/new-{self.smiles_filename}", 
                               delimiter=' ', 
                               titleLine=False)
        
        smiles = [Chem.MolToSmiles(mol) for mol in molecules]
        names = [mol.GetProp("_Name") for mol in molecules]
        standard_smiles = standardize(smiles)

        df = pl.DataFrame(data={"name": names, "smiles": standard_smiles})
        self.dataframe = df
        
    def calculate_fingerprints(self):
        df = self.dataframe
        smiles_list = list(df['smiles'])
        bitstr = getMorganFingerprints(smiles_list)
        fps_array = bitStringToArray(bitstr)
        
        self.fingerprints = fps_array
        return fps_array

    def calculate_mordred(self):
        df = self.dataframe
        smiles_list = list(df['smiles'])
        names = list(df['name'])

        self.descriptor_list = {
            'ATSC2c', 'ATSC8d', 'GATS3c', 'GATS4c', 'GATS5c', 'GATS5d', 'GATS5p',
            'GATS5i', 'BCUTc-1l', 'BCUTdv-1l', 'SsOH', 'ETA_dEpsilon_D',
            'FilterItLogS', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA7', 'PEOE_VSA8',
            'PEOE_VSA9', 'PEOE_VSA12', 'PEOE_VSA13', 'SMR_VSA4', 'SlogP_VSA4',
            'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
            'EState_VSA7', 'EState_VSA8', 'VSA_EState4', 'VSA_EState8',
            'VSA_EState9', 'SLogP', 'JGI2', 'JGI4', 'JGI5', 'JGI6', 'JGI7', 'JGI8',
            'JGI10'
            }
        
        
        df_descriptors = getMordredDescriptors(smiles_list,
                                               self.descriptor_list)
        
        with open(self.STD_SCALER_PATH, 'rb') as model_file:
            std_scaler = pickle.load(model_file)
            
        df_scaled = pd.DataFrame(data=std_scaler.transform(df_descriptors),
                                 columns=df_descriptors.columns)
        
        df_mordred = pd.DataFrame(data={"name": names, "smiles": smiles_list})
        df_mordred = pd.concat([df_mordred, df_scaled],
                               axis=1)
        df_mordred = removeMissingRows(df_mordred)
        
        self.mordred_dataframe = pl.from_pandas(df_mordred)
        
        return df_mordred
    
    def inhA_preprocessing(self):

        df_features = self.mordred_dataframe.to_pandas().iloc[:, 2:]       
        # df_features = convertDtypes(df_features)
        print(df_features)
            
        # float_features, int_features = splitIntFromFloat(df_features)
        
        # df_float = df_features[float_features]
        # df_int = df_features[int_features]
        
        with open(self.STD_SCALER_PATH, 'rb') as model_file:
            std_scaler = pickle.load(model_file)
            
        # with open(self.INT_SCALER_PATH, 'rb') as model_file:
        #     int_scaler = pickle.load(model_file)

        df_scaled = pd.DataFrame(data=std_scaler.transform(df_features),
                                       columns=df_features.columns)
        
        # df_int_scaled = pd.DataFrame(data=int_scaler.transform(df_int),
        #                                columns=df_int.columns)
        
        # df_all_features = pd.concat([df_float_scaled, df_int_scaled], axis=1)
        self.X = df_scaled.values
        
        return df_scaled.values
    
    def get_results(self, predictions, model_name):
        df_pred = pd.DataFrame()
        df_pred['name'] = self.mordred_dataframe['name']
        df_pred['smiles'] = self.mordred_dataframe['smiles']
        df_pred[f'{model_name}_pred'] = predictions
        
        return df_pred