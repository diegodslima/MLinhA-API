import pandas as pd

def fixWrongTypeFeatures(df: pd.DataFrame) -> pd.DataFrame:

  for column in df.columns:
      if df[column].dtype == 'float64' and (df[column] % 1 == 0).all():
          df[column] = df[column].astype(int)

  return df