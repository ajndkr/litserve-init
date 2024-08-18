# litserve-init

demo repository to use [`litserve`](https://lightning.ai/docs/litserve).

## setup

1. create virtual environment of your choice. I chose miniconda:

   ```shell
   conda create -n litserve-init python=3.10
   ```

2. install dependencies:

   ```shell
   pip install -r requirements.txt
   ```

   > Disclaimer: `transformers[torch]` will install the latest `torch` version.
   > If not compatible with your GPU, install the correct version of `torch` and
   > `torchvision` separately.

## benchmark

below are the results of a simple benchmark comparison between FastAPI and
LitServe.

> **system requirements**: i used a 32-core linux machine with 2 NVIDIA GeForce
> RTX 3080 Ti GPUs.

each experiment sends `n` concurrent requests containing a single image for
classification. The image can be found [here](./cats-image.png).

in case of the LitServe application, I set the `max_batch_size` to 32 and
`batch_timeout` to 0.05. This means that LitServe will batch incoming requests
with a maximum batch size of 8 within 50 milliseconds for inference. The FastAPI
application will process all incoming requests separately.

### total runtime (per experiment)

| concurrent requests | FastAPI | LitServe |
| ------------------- | ------- | -------- |
| 10                  | 0.163 s | 0.107 s  |
| 50                  | 0.720 s | 0.342 s  |
| 100                 | 1.409 s | 0.638 s  |
| 200                 | 2.952 s | 1.332 s  |

### average response time (per request)

| concurrent requests | FastAPI                | LitServe             |
| ------------------- | ---------------------- | -------------------- |
| 10                  | 81.604 ± 71.672 ms     | 72.229 ± 12.598 ms   |
| 50                  | 366.057 ± 353.857 ms   | 180.208 ± 87.027 ms  |
| 100                 | 725.106 ± 693.911 ms   | 353.102 ± 190.356 ms |
| 200                 | 1555.120 ± 1342.055 ms | 845.292 ± 287.755 ms |

to run the benchmarks yourself, run `./fastapi-benchmark.sh` and
`./litserve-benchmark.sh` commands on your terminal.

### conclusion

in terms of both total runtime and average response time, LitServe outperforms
FastAPI (near 2x faster). This can be attributed to the fact that LitServe
utilises all available GPUs by default and supports batching of concurrent
requests whereas FastAPI uses only one GPU and does not support batching. There
could be other factors at play as well, but these are the most significant ones
in my opinion.

## takeaways

- since LitServe is built on top of FastAPI, it can be a drop-in replacement for
  FastAPI-based machine learning projects
- as i wanted to test the out-of-the-box performance of both frameworks, both
  the applications are built without optimisation in mind. Even the machine
  learning model I used is a simple pre-trained ResNet50. It is thus likely that
  performance may vary for more complex applications/use-cases as LitServe is
  still a relatively new project
- my favorite features till now: **GPU control** and **batching support**. As
  opposed to the FastAPI application which used only one GPU, LitServe spins up
  a worker per all available devices by default (can be configured further).
  Batching is something very exciting for me as it drastically improves the
  concurrency performance of any machine learning application under heavy load.
