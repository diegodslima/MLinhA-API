from app.classes.Dataset import Dataset
from fastapi import FastAPI, APIRouter, File, UploadFile
from fastapi.openapi.utils import get_openapi
import json
import os
import uuid

app = FastAPI()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="MLinhA API",
        version="1.0.0",
        summary="This API provides a cheminformatics service for predicting pIC50 values from molecular structures encoded in Simplified Molecular Input Line Entry System (SMILES) format. \n" + 
        "Users can upload a file containing SMILES representations of chemical compounds. \n" + 
        "The API then utilizes Mordred, a Python library for molecular descriptor calculation, to generate a set of molecular descriptors for each compound. \n" + 
        "Subsequently, a pre-trained machine learning model is employed to predict the pIC50 values based on the extracted molecular features.",
        description="",
        routes=app.routes,
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "App running."}

@app.post("/inhA_pred", tags=["ML Prediction"])
async def inha_prediction(file: UploadFile = File(...)):
    
    current_directory = os.getcwd()
    temp_directory = f"{current_directory}/temp/"

    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    try:
        
        unique_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        new_filename = f"{unique_id}{file_extension}"
        
        
        with open(f"{temp_directory}/{new_filename}", "wb") as f:
            f.write(file.file.read())
        
        dataset = Dataset(new_filename)
        dataset.create_dataframe()
        dataset.calculate_mordred()
        dataset.mlinha_predict()

        parsed_data = json.loads(dataset.inha_prediction.write_json(row_oriented=True))
        return {"num_mols": dataset.inha_prediction.shape[0],
                "results": parsed_data}
    
    except Exception as e:
        print(e)
        return {"message": "Something went wrong.",
                "error": e}
    
    finally: 
        if os.path.exists(f"{temp_directory}/{new_filename}"):
            os.remove(f"{temp_directory}/{new_filename}")
        else:
            print(f"The file {temp_directory}/{new_filename} does not exist.")

        if os.path.exists(f"{temp_directory}/new-{new_filename}"):
            os.remove(f"{temp_directory}/new-{new_filename}")
        else:
            print(f"The file {temp_directory}/new-{new_filename} does not exist.")

@app.post("/mtb_pred", tags=["In Development"])
async def mtb_prediction():
    return {"message": "Endpoint under construction."}

@app.post("/inhA_KDE", tags=["In Development"])
async def inha_kde():
    return {"message": "Endpoint under construction."}

@app.post("/mtb_KDE", tags=["In Development"])
async def mtb_kde():
    return {"message": "Endpoint under construction."}