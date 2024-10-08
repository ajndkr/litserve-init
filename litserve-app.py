from io import BytesIO

import litserve as ls
import torch
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification


class ResNetLitAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.name = "microsoft/resnet-50"

        self.processor = AutoImageProcessor.from_pretrained(self.name)
        self.model = ResNetForImageClassification.from_pretrained(self.name).to(device)

    def decode_request(self, request):
        image_bytes = bytes.fromhex(request["image_bytes"])
        image = Image.open(BytesIO(image_bytes))
        image_tensor = self.processor(image, return_tensors="pt")["pixel_values"]
        return image_tensor.to(self.device)

    def batch(self, inputs):
        return torch.cat(inputs, dim=0)

    def predict(self, inputs):
        with torch.no_grad():
            logits = self.model(inputs).logits
        return logits

    def encode_response(self, logits):
        predicted_label = logits.argmax(-1).item()
        predicted_class = self.model.config.id2label[predicted_label]
        return {"predicted_label": predicted_label, "predicted_class": predicted_class}


if __name__ == "__main__":
    api = ResNetLitAPI()
    server = ls.LitServer(api, accelerator="gpu", max_batch_size=8, batch_timeout=0.05)
    server.run(port=8000)
