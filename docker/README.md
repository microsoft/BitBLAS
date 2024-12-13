To ease the process of installing all the dependencies, baseline software, and Ladder code, we provide a Dockerfile and a simple guideline to build a Docker image with all of above installed. The Docker image is built on top of Ubuntu 20.04, and it contains all the dependencies required to run the experiments. We only provide the Dockerfile for NVIDIA GPU, and the Dockerfile for AMD GPU will be provided upon request.

```bash
git clone --recursive https://github.com/microsoft/BitBLAS BitBLAS
cd BitBLAS/docker
# build the image, this may take a while (around 10+ minutes on our test machine)
docker build -t bitblas_cuda -f Dockerfile.cu120 .
# run the container
docker run -it --cap-add=SYS_ADMIN --network=host --gpus all --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name bitblas_test bitblas_cuda bash
```
