# Aspect Sentiment Quad Prediction (ASQP) [Updated in August 2024]


We highly recommend you to install the specified version of the following packages to avoid unnecessary troubles:

- transformers==4.44.0
- sentencepiece==0.2.0
- torch==2.4.0

Tested on CUDA Version: 12.2. Pytorch Lightning isn't needed anymore!

## Quick Start

- Set up the environment as described in the above section
- Download the pre-trained T5-base model (you can also use larger versions for better performance depending on the availability of the computation resource), put it under the folder `T5-base`.
  - You can also skip this step and the pre-trained model would be automatically downloaded to the cache in the next step
- Run command `sh run.sh`, which runs the ASQP task on the `rest15` dataset.
- More details can be found in the paper and the help info in the `main.py`.


