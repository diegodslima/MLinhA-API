from fastapi import FastAPI, APIRouter, File, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from fastapi.requests import Request

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import json
import os
import uuid
from pathlib import Path
import logging

from app.classes.Model import Model
from app.classes.Dataset import Dataset


app = FastAPI()

script_dir = os.path.dirname(__file__)
st_abs_file_path = os.path.join(script_dir, "app/static/")
tmlpt_abs_file_path = os.path.join(script_dir, "app/templates/")

app.mount("/static", StaticFiles(directory=st_abs_file_path), name="static")
templates = Jinja2Templates(directory=tmlpt_abs_file_path)


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

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html",
                                      context={
                                          "request": request,
                                          "message": "Hello World!!!"
                                          })

@app.post("/descriptors", tags=["Featurizers"])
async def calc_descriptor(file: UploadFile = File(...)):
    current_directory = Path.cwd()
    temp_directory = current_directory / "temp"

    if not temp_directory.exists():
        temp_directory.mkdir()

    try:
        unique_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        new_filename = f"{unique_id}{file_extension}"

        with open(temp_directory / new_filename, "wb") as f:
            f.write(file.file.read())

        dataset = Dataset(new_filename)
        print(dataset)
        dataset.create_dataframe()
        df_mordred = dataset.calculate_mordred()

        return df_mordred
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {"message": "Something went wrong.", "error": str(e)}

    finally:
        try:
            (temp_directory / new_filename).unlink()
            (temp_directory / ("new-" + str(new_filename))).unlink()
        except FileNotFoundError:
            pass
    

@app.post("/inhA_pred", tags=["ML Prediction"])
async def inha_prediction(file: UploadFile = File(...)):
    current_directory = Path.cwd()
    temp_directory = current_directory / "temp"

    if not temp_directory.exists():
        temp_directory.mkdir()

    try:
        unique_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        new_filename = f"{unique_id}{file_extension}"

        with open(temp_directory / new_filename, "wb") as f:
            f.write(file.file.read())

        dataset = Dataset(new_filename)
        dataset.create_dataframe()
        features = dataset.calculate_mordred()
        # features = dataset.inhA_preprocessing()
        
        inha_svr = Model(model_path='app/models/ml-models/SVR-inhA-mordred.joblib', 
                         module='joblib')
        prediction = inha_svr.model.predict(features.iloc[:,2:])
        results = dataset.get_results(prediction, model_name='inhA')
        
        parsed_data = json.loads(results.to_json())
        return {"num_mols": results.shape[0], "results": parsed_data}

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {"message": "Something went wrong.", "error": str(e)}

    finally:
        try:
            (temp_directory / new_filename).unlink()
            (temp_directory / ("new-" + str(new_filename))).unlink()
        except FileNotFoundError:
            pass


@app.post("/mtb_pred", tags=["ML Prediction"])
async def mtb_prediction(file: UploadFile = File(...)):
    current_directory = Path.cwd()
    temp_directory = current_directory / "temp"

    if not temp_directory.exists():
        temp_directory.mkdir()

    try:
        unique_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        new_filename = f"{unique_id}{file_extension}"

        with open(temp_directory / new_filename, "wb") as f:
            f.write(file.file.read())

        dataset = Dataset(new_filename)
        dataset.create_dataframe()
        features = dataset.calculate_fingerprints()

        mtb_rf = Model(model_path='app/models/ml-models/mtb-A25-novo-logreg-classifier.pkl')
        prediction = mtb_rf.model.predict(features)
        # predict_proba = mtb_rf.model.predict_proba(features)
        results = dataset.get_results(list(prediction), model_name='mtb')
        # results['proba_0'] = [prob[0] for prob in predict_proba]
        # results['proba_1'] = [prob[1] for prob in predict_proba]
        parsed_data = json.loads(results.to_json(orient='records'))
        
        return {"num_mols": results.shape[0], "results": parsed_data}

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {"message": "Something went wrong.", "error": str(e)}

    finally:
        try:
            (temp_directory / new_filename).unlink()
            (temp_directory / ("new-" + str(new_filename))).unlink()
        except FileNotFoundError:
            pass

@app.post("/inhA_KDE", tags=["In Development"])
async def inha_kde():
    return {"message": "Endpoint under construction."}

@app.post("/mtb_KDE", tags=["In Development"])
async def mtb_kde():
    return {"message": "Endpoint under construction."}