import base64
from contextlib import asynccontextmanager
from io import BytesIO

import torch
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification

NAME = "microsoft/resnet-50"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet50_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    resnet50_model["processor"] = AutoImageProcessor.from_pretrained(NAME)
    resnet50_model["model"] = ResNetForImageClassification.from_pretrained(NAME).to(
        DEVICE
    )
    yield
    resnet50_model.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
        image_bytes = body.get("image_bytes")
        if not image_bytes:
            raise HTTPException(status_code=400, detail="image_bytes field is missing")

        image_bytes = bytes.fromhex(image_bytes)
        image = Image.open(BytesIO(image_bytes))
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid image data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    inputs = resnet50_model["processor"](image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = resnet50_model["model"](**inputs).logits

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    predicted_label = logits.argmax(-1).item()
    predicted_class = resnet50_model["model"].config.id2label[predicted_label]

    return {"predicted_label": predicted_label, "predicted_class": predicted_class}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
