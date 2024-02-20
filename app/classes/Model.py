import pickle
from joblib import load

class Model:
    def __init__(self, model_path, module='pickle'):
        self.model_path = model_path
        self.model = None
        
        if module == 'pickle':
            with open(self.model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
                print('model loaded')
        elif module == 'joblib':
            svr = load(model_path) 
            self.model = svr