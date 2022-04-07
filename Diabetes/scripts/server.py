# Build fastapi server
# Import FastAPI
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from classification_model import ClassificationModel
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import io
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import copy

import json
# Import uvicorn
import uvicorn
import os
# Create FastAPI app
app = FastAPI()
# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add GET method with request
@app.get("/models/{model_name}")
async def get_model(model_name: str):
    models_list = os.listdir("./"+model_name+"/models/")
    spec = json.load(open("./"+model_name+"/model.spec.json"))
    # Return JSON response
    return JSONResponse(
        content={"models": models_list,"spec":spec},
        status_code=200,
    )


@app.get("/inference/{model_name}/{model_file}/{input}/{norm}")
async def get_inference(model_name: str, model_file: str, input: str, norm:bool):
    input = [[float(x) for x in input.split(",")]]
    try:
        is_keras = model_file[-2:] == "h5"
        model = ClassificationModel.from_file(
            "./"+model_name+"/models/"+model_file, model_file, "./"+model_name+"/plots/describes.csv", is_keras)
        if norm:
            for inp in input:
                for i in range(len(inp)):
              #      print(model.describes)
                    inp[i] = (inp[i] - model.describes.iloc[3,i+1]) / (model.describes.iloc[7, i+1] - model.describes.iloc[3,i+1])
        # result = model.model.predict(input)
        # Calculate predict probability
        result = []
        try:
            result = model.model.predict_proba(input)
        except:
            result = model.model.predict(input)
        if len(result[0]) == 2:
            result = result.tolist()
            result = [[result[0][1]]]
        else:
            result = result.tolist()
        return JSONResponse(
            content={
                "algorithm": model.model_name,
                "result": result},
            status_code=200,
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )

async def calc_derivative(model_name: str, model_file: str, input: str, delta:float=0.00001, norm:bool=True, denorm:bool=False):
    try:
        is_keras = model_file[-2:] == "h5"
        model = ClassificationModel.from_file(
            "./"+model_name+"/models/"+model_file, model_file, "./"+model_name+"/plots/describes.csv", is_keras)
        if norm:
            for inp in input:
                for i in range(len(inp)):
                #      print(model.describes)
                    inp[i] = (inp[i] - model.describes.iloc[3,i+1]) / (model.describes.iloc[7, i+1] - model.describes.iloc[3,i+1])
        denorm_input = input[0].copy()
        if denorm:
            for i in range(len(input[0])):
                denorm_input[i] = denorm_input[i] * (model.describes.iloc[7, i+1] - model.describes.iloc[3,i+1]) + model.describes.iloc[3,i+1]
        # result = model.model.predict(input)
        # Calculate predict probability
        result = []
        upper_deri = []
        lower_deri = []
        try:
            result = model.model.predict_proba(input)
        except:
            result = model.model.predict(input)
        if len(result[0]) == 2:
            result = result.tolist()
            result = [[result[0][1]]]
        else:
            result = result.tolist()
        for i in range(len(input[0])):
            inp = input.copy()
            inp[0][i] = inp[0][i] + delta
            ud = 0
            ld = 0
            try:
                ud = (model.model.predict_proba(inp)[0][1] - result[0][0]) / delta
                ld = (-model.model.predict_proba(inp)[0][1] + result[0][0]) / (-delta)
            except:
                ud = (model.model.predict(inp)[0][0] - result[0][0]) / delta
                ld = (-model.model.predict(inp)[0][0] + result[0][0]) / (-delta)
            upper_deri.append(ud)
            lower_deri.append(ld)
        
        avg_deri = []
        for i in range(len(upper_deri)):
            avg_deri.append((upper_deri[i] + lower_deri[i]) / 2)
        return (None,upper_deri,lower_deri, avg_deri,result,denorm_input)
    except Exception as e:
        return (e,None,None,None,None,[])


@app.get("/derivative/{model_name}/{model_file}/{input}/{selected_features}/{delta}/{lr}/{step}/{norm}")
async def get_derivative(model_name: str, model_file: str, input: str, selected_features:str, delta:float=0.00001,lr:float=0.01,step:int=100, norm:bool=True):
        input = [[float(x) for x in input.split(",")]]
        selected_features = [True if x == "1" else False for x in selected_features.split(",")]
        if len(selected_features) != len(input[0]):
            while len(selected_features) != len(input[0]):
                selected_features.append(True)
        err, upper_deri, lower_deri, deri, result, _ = await calc_derivative(model_name, model_file, input, delta, norm)
        # Calculate gradient descent for each features
        updated_x =  input.copy()
        updated_deri = deri.copy()
        history = []
        updated_result = []
        for k in range(step):
            for i in range(len(deri)):
                if selected_features[i]:
                    updated_x[0][i] -= lr * updated_deri[i]
            err,_,_,updated_deri,updated_result, denorm_x = await calc_derivative(model_name, model_file, updated_x, delta, False, True)
            if err == None:
                history.append({"step":k,"result":updated_result, "x":denorm_x})
        if err:
            return JSONResponse(
            content={"error": str(err)},
            status_code=500,
        )
        return JSONResponse(
            content={
                "algorithm": model_name,
                "upper_derivative": upper_deri,
                "lower_derivative": lower_deri,
                "derivative": deri,
                "gradient_descent": history,
                "result": result},
            status_code=200,
        )
       

@app.get("/derivative-by-delta-plot/{model_name}/{model_file}/{input}/{delta_lower}/{delta_upper}/{step}/{norm}")
async def derivative_plot(model_name: str, model_file: str, input: str, delta_lower:float=0.1,delta_upper:float=2,step:float=0.1, norm:bool=True):
    input = [[float(x) for x in input.split(",")]]
    delta = delta_lower
    delta_list = []
    deri_list = []
    while delta < delta_upper:
        err, upper_deri, lower_deri, deri, result,_ = await calc_derivative(model_name, model_file, input, delta, norm)
        if err:
            return JSONResponse(
            content={"error": str(err)},
            status_code=500,
        )
        delta_list.append(delta)
        deri_list.append(deri)
        delta += step
    # Plot derivative by delta
    # clear plot
    plt.clf()
    plt.plot(delta_list, deri_list)
    plt.xlabel('Delta')
    plt.ylabel('Derivative')
    plt.title('Derivative by delta')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")