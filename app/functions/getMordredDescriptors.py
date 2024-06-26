import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

def getMordredDescriptors(smiles_list, descriptor_list):

    calc = Calculator(descriptors, ignore_3D=True)
    calc.descriptors = [d for d in calc.descriptors if str(d) in descriptor_list]

    print("Number of features selected: ", len(calc.descriptors))
    
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    df = calc.pandas(mols)

    return df