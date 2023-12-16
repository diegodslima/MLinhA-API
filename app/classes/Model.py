import pickle

class Model:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        
        with open(self.model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
            print('model loaded')