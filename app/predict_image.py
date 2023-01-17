from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse
import cv2
import torch
import numpy as np
from src.models.model import ResNeSt
import hydra
from hydra import compose
from pydantic import BaseModel
from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


# set up tracing and open telemetry
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResponseModel(BaseModel):
    filename: str
    label: int


@app.get("/")
async def root():
    return RedirectResponse("/docs")

@app.post("/predict/", response_model=ResponseModel)
async def predict(data: UploadFile = File(...)):
    with open('temp.jpg', 'wb') as image:
        logger.info("Received image file")
        filename = data.filename
        content = await data.read()
        image.write(content)
        image.close()
    
    top_class = predict()
    logger.info("Made prediction")

    response = ResponseModel(filename=filename, label=top_class)
    logger.info("Built response")

    return response

def predict():
    hydra.initialize(config_path="../conf", job_name="predict")
    config = compose(config_name='predict.yaml')

    image_path = 'temp.jpg'
    top_class = predict_step(config, image_path)

    return top_class

def predict_step(config, image_path):
    model = ResNeSt.load_from_checkpoint("models/model.ckpt", map_location=device, hparams=config)
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))

    MEAN = 255 * np.array([0.485, 0.456, 0.406])
    STD = 255 * np.array([0.229, 0.224, 0.225])

    img = (img - MEAN) / STD
    img = img.T
    img = img[np.newaxis, :, :, :]

    img = torch.from_numpy(img).float()
    img = img.to(device)

    y_pred = model.forward(img)
    ps = torch.exp(y_pred)
    _, top_class = ps.topk(1, dim=1)

    return top_class.item()
