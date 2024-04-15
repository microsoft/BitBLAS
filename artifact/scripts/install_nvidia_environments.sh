# !/bin/bash

# python requirements
pip install prettytable

# install torch inductor
pip install torch>=2.0.0

# install onnxruntime
pip install onnx

pip install onnxruntime-gpu==1.16.1

# install tensor ir
./scripts/install_tensorir.sh

# install amos
./scripts/install_amos.sh

# install roller
./scripts/install_roller.sh

# install welder
./scripts/install_welder.sh

# install vLLM
./scripts/install_vllm.sh

# install tenosrrt
./scripts/install_tensorrt.sh
