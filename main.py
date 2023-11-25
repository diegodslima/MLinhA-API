from src.classes.Dataset import Dataset
from fastapi import FastAPI, File, UploadFile
import json

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "App running."}

@app.post("/inhA_pred")
async def inha_prediction(file: UploadFile = File(...)):

    with open(f"temp/{file.filename}", "wb") as f:
        f.write(file.file.read())

    file_path = f"temp/{file.filename}"

    dataset = Dataset(file_path)
    dataset.create_dataframe()
    dataset.calculate_mordred()
    dataset.mlinha_predict()
    
    parsed_data = json.loads(dataset.inha_prediction.write_json(row_oriented=True))
    
    return parsed_data