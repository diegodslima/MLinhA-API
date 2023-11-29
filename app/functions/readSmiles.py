from rdkit import Chem

def readSmiles(path, delimiter, titleLine):
    
    with Chem.SmilesMolSupplier(path, delimiter=delimiter,  titleLine=titleLine) as supplier:
        molecules = [mol for mol in supplier]
        
        return molecules