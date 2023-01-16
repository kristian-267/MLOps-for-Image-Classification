from fastapi import FastAPI, UploadFile, File
import cv2
import torch
import numpy as np
from src.models.model import ResNeSt
import hydra
from hydra import compose


app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict():
    hydra.initialize(config_path="conf", job_name="prediction_app")
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

@app.post("/")
async def predict_image(data: UploadFile = File(...)):
    with open('temp.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)
        image.close()
    
    top_class = predict()

    response = f"The Image Is Belongs To Class {top_class}."

    return response
