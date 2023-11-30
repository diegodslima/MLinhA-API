from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.MolStandardize import rdMolStandardize

def standardize(smiles_list):
    RDLogger.DisableLog('rdApp.*')
    uncharger = rdMolStandardize.Uncharger()

    processed_mols = []

    for smiles in smiles_list:
        mol = MolFromSmiles(smiles)
        parent_clean_mol = rdMolStandardize.FragmentParent(mol)
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        processed_mols.append(MolToSmiles(uncharged_parent_clean_mol))

    return processed_mols