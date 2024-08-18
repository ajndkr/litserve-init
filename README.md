# litserve-init

Demo repository to use [`litserve`](https://lightning.ai/docs/litserve).

## setup

1. create virtual environment of your choice. I chose miniconda:

   ```shell
   conda create -n litserve-init python=3.10
   ```

1. install dependencies:

   ```shell
   pip install -r requirements.txt
   ```

   > Disclaimer: `transformers[torch]` will install the latest `torch` version. If not compatible
   > with your GPU, install the correct version of `torch` and `torchvision` separately.
