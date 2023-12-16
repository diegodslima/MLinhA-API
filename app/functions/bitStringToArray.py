import polars as pl
import numpy as np

def bitStringToArray(fingerprints):
        
    fps_array = np.array([list(s) for s in fingerprints], dtype=np.int32)
    
    return fps_array