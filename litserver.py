import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from io import BytesIO
from PIL import Image
import litserve as ls


class ResNetLitAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.name = "microsoft/resnet-50"

        self.processor = AutoImageProcessor.from_pretrained(self.name)
        self.model = ResNetForImageClassification.from_pretrained(self.name).to(device)

    def decode_request(self, request):
        image_bytes = bytes.fromhex(request["image_bytes"])
        image = Image.open(BytesIO(image_bytes))
        return self.processor(image, return_tensors="pt").to(self.device)

    def predict(self, inputs):
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits

    def encode_response(self, logits):
        predicted_label = logits.argmax(-1).item()
        predicted_class = self.model.config.id2label[predicted_label]
        return {"predicted_label": predicted_label, "predicted_class": predicted_class}


if __name__ == "__main__":
    api = ResNetLitAPI()
    server = ls.LitServer(api, accelerator="gpu")
    server.run(port=8000)
