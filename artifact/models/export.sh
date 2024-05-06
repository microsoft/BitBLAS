# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


pip install einops
pip install timm
pip install torch>=2.0.0
pip install onnxconverter_common
pip install transformers==4.35.0

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

echo "Exporting BLOOM model with batchsize 1 and seq 1"
cd bloom_176b
python gen.py --batch_size 1 --seq_length 1
cd ..
echo "Exporting BLOOM model with batchsize 1 and seq 1 done."

echo "Exporting BLOOM model with batchsize 32 and seq 1"
cd bloom_176b
python gen.py --batch_size 32 --seq_length 1
cd ..
echo "Exporting BLOOM model with batchsize 32 and seq 1 done."

echo "Exporting BLOOM model with batchsize 1 and seq 4096"
cd bloom_176b
python gen.py --batch_size 1 --seq_length 4096
cd ..
echo "Exporting BLOOM model with batchsize 1 and seq 4096 done."

echo "Exporting LLAMA model with batchsize 1 and seq 1"
cd llama_70b
python gen.py --batch_size 1 --seq_length 1
cd ..
echo "Exporting LLAMA model with batchsize 1 and seq 1 done."

echo "Exporting LLAMA model with batchsize 32 and seq 1"
cd llama_70b
python gen.py --batch_size 32 --seq_length 1
cd ..
echo "Exporting LLAMA model with batchsize 32 and seq 1 done."


echo "Exporting LLAMA model with batchsize 1 and seq 4096"
cd llama_70b
python gen.py --batch_size 1 --seq_length 4096
cd ..
echo "Exporting LLAMA model with batchsize 1 and seq 4096 done."
