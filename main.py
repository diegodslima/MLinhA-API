from src.classes.Dataset import Dataset
        
smiles_path = "src/dataset/example/best_ligands.smi"
dataset = Dataset(smiles_path)

dataset.create_dataframe()
dataset.calculate_mordred()
dataset.mlinha_predict()

print(dataset.inha_prediction)