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
    MLP_MODEL_PATH = 'app/models/ml-models/mlp_inha_model.pkl'
    STD_SCALER_PATH = 'app/models/scalers/std-scaler-inhA-small-nov23.pkl'
    INT_SCALER_PATH = 'app/models/scalers/int-scaler-inhA-small-nov23.pkl'

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
            'AATS6m', 'ATSC1dv', 'SssCH2', 'SsssCH', 'SaasN', 'SdO', 'PEOE_VSA1',
             'SMR_VSA3', 'SlogP_VSA5', 'EState_VSA8', 'VSA_EState2', 'MID_N',
             'TopoPSA(NO)', 'TopoPSA', 'GGI4', 'SRW07', 'SRW09', 'TSRW10',
             'nAromAtom', 'nAromBond', 'nBondsA', 'C1SP2', 'n5aRing', 'n5aHRing'
            }

        dataset = {"name": names, "smiles": smiles_list}
        df_mordred = pd.DataFrame(data=dataset)

        df_mordred = pd.concat([df_mordred, getMordredDescriptors(smiles_list, 
                                                                  self.descriptor_list)], axis=1)

        df_mordred = removeMissingRows(df_mordred)
        self.mordred_dataframe = pl.from_pandas(df_mordred)

    def inhA_preprocessing(self):

        df_features = self.mordred_dataframe.to_pandas().iloc[:, 2:]       
        df_features = convertDtypes(df_features)
            
        float_features, int_features = splitIntFromFloat(df_features)
        
        df_float = df_features[float_features]
        df_int = df_features[int_features]
        
        with open(self.STD_SCALER_PATH, 'rb') as model_file:
            std_scaler = pickle.load(model_file)
            
        with open(self.INT_SCALER_PATH, 'rb') as model_file:
            int_scaler = pickle.load(model_file)

        df_float_scaled = pd.DataFrame(data=std_scaler.transform(df_float),
                                       columns=df_float.columns)
        
        df_int_scaled = pd.DataFrame(data=int_scaler.transform(df_int),
                                       columns=df_int.columns)
        
        df_all_features = pd.concat([df_float_scaled, df_int_scaled], axis=1)
        self.X = df_all_features.values
        
        return df_all_features.values
    
    def get_results(self, predictions, model_name):
        df_pred = pd.DataFrame()
        df_pred['name'] = self.dataframe['name']
        df_pred['smiles'] = self.dataframe['smiles']
        df_pred[f'{model_name}_pred'] = predictions
        
        return df_pred