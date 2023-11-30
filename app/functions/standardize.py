from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

# Disable RDKit's informational messages
RDLogger.DisableLog('rdApp.*')

uncharger = rdMolStandardize.Uncharger()

def standardize(smiles_list):
    processed_mols = [None] * len(smiles_list)

    for i, smiles in enumerate(smiles_list):
        mol = MolFromSmiles(smiles)
        # clean_mol = rdMolStandardize.Cleanup(mol)
        parent_clean_mol = rdMolStandardize.FragmentParent(mol)
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        # taut_uncharged_parent_clean_mol = taut_enum.Canonicalize(uncharged_parent_clean_mol)
        processed_mols[i] = MolToSmiles(uncharged_parent_clean_mol)
        
    return processed_mols