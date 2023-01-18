import cv2
import numpy as np
import torch
from google.cloud import storage

MODEL_BUCKET_NAME = "dtu-mlops-group7-models"
MODEL_FILE = "deployable_model.pt"
IMAGE_BUCKET_NAME = "dtu-mlops-group7-images"

client = storage.Client()
model_bucket = client.get_bucket(MODEL_BUCKET_NAME)
model_blob = model_bucket.get_blob(MODEL_FILE)
model_blob.download_to_filename("/tmp/model.pt")
model = torch.jit.load("/tmp/model.pt")
model.eval()


def resnest_classifier(request):
    """will to stuff to your request"""
    request_json = request.get_json()
    results = {}
    if request_json and "operation" in request_json:
        if request_json["operation"] == "predict":
            image_blobs = client.list_blobs(IMAGE_BUCKET_NAME)
            for image_blob in image_blobs:
                if image_blob.name != IMAGE_BUCKET_NAME:
                    image_blob.download_to_filename("/tmp/image.jpg")
                    img = cv2.imread("/tmp/image.jpg", cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (224, 224))

                    MEAN = 255 * np.array([0.485, 0.456, 0.406])
                    STD = 255 * np.array([0.229, 0.224, 0.225])

                    img = (img - MEAN) / STD
                    img = img.T
                    img = img[np.newaxis, :, :, :]

                    img = torch.from_numpy(img).float()

                    y_pred = model.forward(img)
                    ps = torch.exp(y_pred)
                    _, top_class = ps.topk(1, dim=1)

                    results.update(
                        {
                            "filename": image_blob.name,
                            "label": top_class.item(),
                        }
                    )
        else:
            return "No such operation"
    else:
        return "Request invalid"

    return results
