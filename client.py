import httpx
from PIL import Image
from io import BytesIO


def main():
    image = Image.open("cats-image.png").convert("RGB")

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    response = httpx.post(
        "http://localhost:8000/predict",
        json={
            "image_bytes": image_bytes.hex()
        },
    )

    if response.status_code != 200:
        print(f"Error: (code={response.status_code}, message={response.text})")
        return

    print("output:", response.json())


if __name__ == "__main__":
    main()
