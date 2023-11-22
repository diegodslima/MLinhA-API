import pickle

def standardScaler(scaler_path, X):

    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
        
    X_scaled = scaler.transform(X)
    
    return X_scaled