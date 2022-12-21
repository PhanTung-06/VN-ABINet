from distutils.command.config import config
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import shutil
import gdown 
import torch


from utils.detector import Detector
from detectron2.data.detection_utils import read_image
import torch as t
print(t.cuda.is_available())
# url_model = "https://drive.google.com/u/0/uc?id=15AI1jK6jQkxClQoClohajXW_CjtYBWYl&export=download"
# output = "trained_model.pth"
# gdown.download(url_model, output)
detector = Detector('./configs/ABINet/VinText.yaml', './model_0059999.pth')
# detector = Detector('./config/BAText/TotalText/attn_R_50.yaml','model_tt_att.pth')

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request
    })


@app.post("/", response_class=HTMLResponse)
async def create_upload_files(request: Request, image: UploadFile = File(...)):
    with open("./static/images/test.jpg", "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    img = read_image('./static/images/test.jpg', format='BGR')
    prediction, vis = detector.predict(img)
    vis.save("./static/images/output.jpg")

    return templates.TemplateResponse("result.html", {
        "request": request
    })
        


