import polars as pl

df = pl.read_parquet('./best_ligands_mordred_predictions.parquet')
print(df)