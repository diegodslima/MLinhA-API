from src.classes.Dataset import Dataset
from fastapi import FastAPI, File, UploadFile
import json
import os
import uuid

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "App running."}

@app.post("/inhA_pred")
async def inha_prediction(file: UploadFile = File(...)):
    try:
        
        unique_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        new_filename = f"{unique_id}{file_extension}"
        print(new_filename)

        with open(f"temp/{new_filename}", "wb") as f:
            f.write(file.file.read())

        file_path = f"temp/{new_filename}"

        dataset = Dataset(file_path)
        dataset.create_dataframe()
        dataset.calculate_mordred()
        dataset.mlinha_predict()
        
        parsed_data = json.loads(dataset.inha_prediction.write_json(row_oriented=True))
        
        return parsed_data
    
    finally:
        os.remove(file_path)