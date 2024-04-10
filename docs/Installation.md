# Installation Guide

## Prerequisites

- **Operating System**: Linux
- **Python Version**: >= 3.7
- **CUDA Version**: >= 10.0

## Installing with pip

The easiest way to install BitBLAS is direcly from the PyPi using pip. To install the latest version, run the following command in your terminal.
```bash
pip install bitblas
```

Alternatively, you may choose to install BitBLAS using prebuilt packages available on the Release Page:

```bash
pip install bitblas-0.0.0.dev0+cu120-py3-none-any.whl
```

After installing BitBLAS, you can verify the installation by running:

```bash
python -c "import bitblas; print(bitblas.__version__)"  
```

## Building from Source

We recommend using a docker container with the necessary dependencies to build BitBLAS from source. You can use the following command to run a docker container with the necessary dependencies:

```bash
docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.01-py3
```

To build and install BitBLAS directly from source, follow the steps below. This process requires certain pre-requisites from apache tvm, which can be installed on Ubuntu/Debian-based systems using the following commands:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

After installing the prerequisites, you can clone the BitBLAS repository and install it using pip:

```bash
git clone --recursive https://github.com/Microsoft/BitBLAS.git
cd BitBLAS
pip install -e .  # Please be patient, this may take some time.
```
