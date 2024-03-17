# hf-codecomplete-server - Starcoder2 support

Under progress:
    starcoder2 awq

## System Requirements

- Python 3.11
- Nvidia GPU (for running the model on the server) - Only tested 20 series and above

## Installation and Usage

1. Install the required dependencies

        pip install -r requirements.txt

2. Launch server

       python3 server.py

   This should launch the instance at the default port used by the HF extension. The vLLM library will download the model and perform transformations.
