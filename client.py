import asyncio
import time
from io import BytesIO

import click
import httpx
import pandas as pd
from PIL import Image


async def post(
    client: httpx.AsyncClient, url: str, image_bytes: bytes
) -> tuple[str, float]:
    start_time = time.time()
    while True:
        try:
            response = await client.post(
                url,
                json={"image_bytes": image_bytes.hex()},
            )

            if response.status_code != 200:
                print(f"Error: (code={response.status_code}, message={response.text})")
                continue

            return response.json(), start_time, time.time()
        except Exception as e:
            print("error: ", e)
            continue


async def run_predictions(
    client: httpx.AsyncClient, url: str, n: int, image_bytes: bytes
) -> list[tuple[str, float]]:
    tasks = [post(client, url, image_bytes) for _ in range(n)]
    return await asyncio.gather(*tasks)


async def run(url: str, n_jobs: int, image_bytes: bytes) -> pd.DataFrame:
    results = []

    async with httpx.AsyncClient() as client:
        predictions = await run_predictions(client, url, n_jobs, image_bytes)
        results.extend(
            [
                dict(
                    job_id=job_id,
                    predicted_label=prediction["predicted_label"],
                    predicted_class=prediction["predicted_class"],
                    start_time=pd.to_datetime(start_time, unit="s"),
                    end_time=pd.to_datetime(end_time, unit="s"),
                    elapsed_time=(end_time - start_time),
                )
                for job_id, (prediction, start_time, end_time) in enumerate(predictions)
            ]
        )

    return pd.DataFrame(results)


@click.command()
@click.option("--infile", "infile", type=click.Path(), required=True)
@click.option("--url", "url", type=str, default="http://localhost:8000/predict")
@click.option("--jobs", "n_jobs", type=int, default=1)
@click.option("--outfile", "outfile", type=click.Path(), default="results.csv")
def main(
    infile: str,
    url: str,
    n_jobs: int,
    outfile: str,
):
    print("configuration:")
    print(f"- concurrent jobs: {n_jobs}")

    image = Image.open("cats-image.png").convert("RGB")

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    df = asyncio.run(run(url, n_jobs, image_bytes))
    df.to_csv(outfile, index=False)

    print("results:")
    print(df)


if __name__ == "__main__":
    main()
