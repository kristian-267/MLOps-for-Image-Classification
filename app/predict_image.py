from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse
import json
import cv2
import torch
import numpy as np
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
    label: str


@app.get("/")
async def root():
    return RedirectResponse("/docs")

@app.post("/predict/", response_model=ResponseModel)
async def predict(data: UploadFile = File(...)):
    with open('tmp/temp.jpg', 'wb') as image:
        logger.info("Received image file")
        filename = data.filename
        content = await data.read()
        image.write(content)
        image.close()
    
    label = predict()
    logger.info("Made prediction")

    response = ResponseModel(filename=filename, label=label)
    logger.info("Built response")

    return response

def predict():
    model = torch.jit.load('model_store/deployable_model.pt')
    model.eval()

    img = cv2.imread('/tmp/temp.jpg', cv2.IMREAD_COLOR)
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

    label = mapping_to_label(top_class.item())

    return label

def mapping_to_label(top_class):
    with open('app/index_to_name.json') as f:
        data = json.load(f)
        f.close()
    
    label = data[str(top_class)][1]
    
    return label
