import pandas as pd

def splitIntFromFloat(df: pd.DataFrame) -> list:

  float_features, int_features = [],[]

  for feature in df.columns:
    feature_type = df[feature].dtype
    if feature_type == 'float64':
      float_features.append(feature)
    elif feature_type == 'int64':
      int_features.append(feature)

  return float_features, int_features