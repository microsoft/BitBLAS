# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# python requirements
pip install prettytable
pip install matplotlib

# install torch inductor
pip install torch>=2.0.0

# install onnxruntime
pip install onnx

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

# install faster transformer byoc
./scripts/install_faster_transformer_tvm.sh
