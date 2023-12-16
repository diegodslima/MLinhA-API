import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# tqdm._instances.clear()

def getMorganFingerprints(smiles_list, n_bits=1024, radius=2, use_features=False):
    fp_list = []
    
    with tqdm(total=len(smiles_list), desc="Calculating Fingerprints") as pbar:
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useFeatures=use_features).ToBitString()
            fp_list.append(fp)
            pbar.update(1)

    return fp_list
