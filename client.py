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
    output = {}
    start_time = time.time()

    try:
        response = await client.post(
            url,
            json={"image_bytes": image_bytes.hex()},
        )

        if response.status_code != 200:
            print(f"error: (code={response.status_code}, message={response.text})")
        else:
            output = response.json()

    except Exception as e:
        print(f"error: (message={str(e)})")

    return output, start_time, time.time()


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
                    predicted_label=prediction.get("predicted_label", None),
                    predicted_class=prediction.get("predicted_class", None),
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

    start_time = time.time()

    df = asyncio.run(run(url, n_jobs, image_bytes))

    run_time = time.time() - start_time

    df.to_csv(outfile, index=False)

    n_success = df.predicted_label.notna().sum()
    n_error = df.predicted_label.isna().sum()

    print("results:")
    print(f"- total runtime: {run_time:.3f} s")
    print(f"- success: {n_success}")
    print(f"- error: {n_error}")

    if n_success > 0:
        mean_elapsed_time_ms = df[df.predicted_label.notna()].elapsed_time.mean() * 1000
        std_elapsed_time_ms = df[df.predicted_label.notna()].elapsed_time.std() * 1000
        print(f"- avg elapsed time: {mean_elapsed_time_ms:.3f} ± {2 * std_elapsed_time_ms:.3f} ms")


if __name__ == "__main__":
    main()
