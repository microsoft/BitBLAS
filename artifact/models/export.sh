# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# !/bin/bash

pip install einops
pip install timm
pip install torch>=2.0.0

echo "Converting ResNet-50 with batch size 128..."
python torch2onnx.py --bs 128 --prefix resnet-50-b128 --fp16 resnet50
echo "ResNet-50 conversion done."

echo "Converting ShuffleNet with batch size 128..."
python torch2onnx.py --bs 128 --prefix shufflenet-b128 --fp16 shufflenet
echo "ShuffleNet conversion done."

echo "Converting Conformer with batch size 128..."
python torch2onnx.py --bs 128 --prefix Conformer-b128 --fp16 Conformer
echo "Conformer conversion done."

echo "Converting Vision Transformer (ViT) with batch size 128..."
python torch2onnx.py --bs 128 --prefix vit-b128 --fp16 vit
echo "Vision Transformer conversion done."

echo "Converting ResNet-50 with batch size 1..."
python torch2onnx.py --bs 1 --prefix resnet-50-b1 --fp16 resnet50
echo "ResNet-50 batch size 1 conversion done."

echo "Converting ShuffleNet with batch size 1..."
python torch2onnx.py --bs 1 --prefix shufflenet-b1 --fp16 shufflenet
echo "ShuffleNet batch size 1 conversion done."

echo "Converting Conformer with batch size 1..."
python torch2onnx.py --bs 1 --prefix Conformer-b1 --fp16 Conformer
echo "Conformer batch size 1 conversion done."

echo "Converting Vision Transformer (ViT) with batch size 1..."
python torch2onnx.py --bs 1 --prefix vit-b1 --fp16 vit
echo "Vision Transformer batch size 1 conversion done."
