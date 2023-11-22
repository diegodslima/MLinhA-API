from src.classes.Dataset import Dataset
        
# Example Usage:
smiles_path = "src/dataset/example/best_ligands.smi"
best_ligands_dataset = Dataset(smiles_path)

best_ligands_dataset.create_dataframe()
best_ligands_dataset.calculate_mordred()
best_ligands_dataset.mlinha_predict()
best_ligands_dataset.print_inhA_predictions()