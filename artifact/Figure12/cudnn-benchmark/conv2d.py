import torch
import time
import numpy as np
import argparse

def conv2d_cuda(N, C, H, W, K, R, S, stride, padding, dilation, dtype):
    A_np = np.random.uniform(-10, 10, [N, C, H, W]).astype("float32")
    B_np = np.random.uniform(-10, 10, [K, C, R, S]).astype("float32")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if dtype == "FP16":  # HMMA-16, torch.float16 or torch.half
        A_torch = torch.tensor(A_np).type(torch.float16).cuda()
        B_torch = torch.tensor(B_np).type(torch.float16).cuda()
    elif dtype == "BF16":  # HMMA-16, only on NVIDIA A100, torch.bfloat16
        A_torch = torch.tensor(A_np).type(torch.bfloat16).cuda()
        B_torch = torch.tensor(B_np).type(torch.bfloat16).cuda()
    elif dtype == "FP32":
        A_torch = torch.tensor(A_np).type(torch.float32).cuda()
        B_torch = torch.tensor(B_np).type(torch.float32).cuda()
    elif dtype == "TF32":  # HMMA-19, NVIDIA A100
        # Please upgrade torch to 1.7; only supported on A100
        # https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        A_torch = torch.tensor(A_np).type(torch.float32).cuda()
        B_torch = torch.tensor(B_np).type(torch.float32).cuda()
    elif dtype == "INT8":  # IMMA, but pytorch has no support for INT8 GEMM
        A_torch = torch.tensor(A_np).type(torch.int8).cuda()
        B_torch = torch.tensor(B_np).type(torch.int8).cuda()
    # Pytorch has no int4 type
    elif dtype == "BOOL":  # BMMA, but pytorch has no support for GEMM GEMM
        A_torch = torch.tensor(A_np).type(torch.bool).cuda()
        B_torch = torch.tensor(B_np).type(torch.bool).cuda()
    elif dtype == "FP64":  # DMMA(FP64), only supported on A100
        A_torch = torch.tensor(A_np).type(torch.float64).cuda()
        B_torch = torch.tensor(B_np).type(torch.float64).cuda()
    else:
        assert False, "wrong type: " + dtype

    global RUN_NUMBER
    number, repeats = RUN_NUMBER

    for i in range(repeats):
        time_record = []
        for j in range(number):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            C_torch = torch.nn.functional.conv2d(
                A_torch, B_torch, bias=None, stride=stride, padding=padding, dilation=dilation
            )

            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
        if i == repeats - 1:
            mean_cost = np.mean(time_record)
    # print("conv2d_cuda, dtype = %s, A: %s, B: %s, C:%s" % (dtype, A_torch.dtype, B_torch.dtype, C_torch.dtype))
    # print(",".join(map(str, [N, C, H, W, K, R, S, stride, padding, dilation, dtype, mean_cost])))
    print(mean_cost)


def mixed_precision_conv2d_cuda(N, C, H, W, K, R, S, stride, padding, dilation, dtype):
    torch.cuda.amp.autocast()
    A_np = np.random.uniform(-10, 10, [N, C, H, W]).astype("float32")
    B_np = np.random.uniform(-10, 10, [K, C, R, S]).astype("float32")

    if dtype == "FP16":  # HMMA-16, torch.float16 or torch.half
        A_torch = torch.tensor(A_np).type(torch.float32).cuda()
        B_torch = torch.tensor(B_np).type(torch.float32).cuda()
    else:
        assert False, "wrong type: " + dtype

    global RUN_NUMBER
    number, repeats = RUN_NUMBER

    for i in range(repeats):
        time_record = []
        with torch.cuda.amp.autocast():
            for j in range(number):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                C_torch = torch.nn.functional.conv2d(
                    A_torch, B_torch, bias=None, stride=stride, padding=padding, dilation=dilation
                )

                end.record()
                torch.cuda.synchronize()
                total = start.elapsed_time(end)
                time_record.append(total)
            if i == repeats - 1:
                mean_cost = np.mean(time_record)
    # print("conv2d_cuda, dtype = %s, A: %s, B: %s, C:%s" % (dtype, A_torch.dtype, B_torch.dtype, C_torch.dtype))
    # print(",".join(map(str, [N, C, H, W, K, R, S, stride, padding, dilation, dtype, mean_cost])))
    print(mean_cost)


def conv2d_llvm(N, C, H, W, K, R, S, stride, padding, dilation, dtype):
    torch.cuda.amp.autocast()
    A_np = np.random.uniform(-10, 10, [N, C, H, W]).astype("float32")
    B_np = np.random.uniform(-10, 10, [K, C, R, S]).astype("float32")

    if dtype == "INT8":
        A_torch = torch.tensor(A_np).type(torch.int8)
        B_torch = torch.tensor(B_np).type(torch.int8)
    else:
        assert False, "wrong type: " + dtype

    global RUN_NUMBER
    number, repeats = RUN_NUMBER

    for i in range(repeats):
        time_record = []
        for j in range(number):
            beg = time.time()

            C_torch = torch.nn.functional.conv2d(
                A_torch, B_torch, bias=None, stride=stride, padding=padding, dilation=dilation
            )

            time_record.append((time.time() - beg) * 1e3)
        if i == repeats - 1:
            mean_cost = np.mean(time_record)
    print(
        "conv2d_cuda, dtype = %s, A: %s, B: %s, C:%s"
        % (dtype, A_torch.dtype, B_torch.dtype, C_torch.dtype)
    )
    print(",".join(map(str, [N, C, H, W, K, R, S, stride, padding, dilation, dtype, mean_cost])))
    # print(mean_cost)


def run_cuda():
    global RUN_CONFIG
    batch, beg, num, dtype = RUN_CONFIG
    print("N, C, H, W, K, R, S, stride, padding, dilation, type, cost")
    
    costs = []
    for i, shape in enumerate(benchmark_shapes[beg : beg + num]):
        (N, C, H, W, K, _, R, S, _, stride, padding, dilation, _) = shape
        if padding == 'SAME':
            assert(stride == 1, "stride must be 1 when padding is SAME")
            padding = (R - 1) // 2
        elif padding == 'VALID':
            padding = 0
        conv2d_cuda(N, C, H, W, K, R, S, stride, padding, dilation, dtype)
        # mixed_precision_conv2d_cuda(N, C, H, W, K, R, S, stride, padding, dilation, dtype)
    print("cudnn: %s" % ("enabled" if torch.backends.cudnn.enabled else "disabled"))


def run_llvm():
    global RUN_CONFIG
    batch, beg, num, dtype = RUN_CONFIG
    print("N, C, H, W, K, R, S, stride, padding, dilation, type, cost")
    
    costs = []
    for i, shape in enumerate(benchmark_shapes[beg : beg + num]):
        (N, C, H, W, K, _, R, S, _, stride, padding, dilation, _) = shape
        # N = batch
        if padding == 'SAME':
            padding = (stride * (W - 1) + S - stride) / 2
            # padding = (R - 1) // 2
        elif padding == 'VALID':
            padding = 0
        conv2d_llvm(N, C, H, W, K, R, S, stride, padding, dilation, dtype)
        # mixed_precision_conv2d_llvm(N, C, H, W, K, R, S, stride, padding, dilation, dtype)

# benchmark_shapes = [
#     # resnet-18
#     # (N, C, H, W, K, _, R, S, _, stride, padding, dilation, _)
#     # (N, C, H, W, F, -1, K, K, -1, S, P, D, -1),
# (128, 512, 7, 7, 2048, -1, 1, 1, -1, 1, 0, 1, -1),
# (128, 512, 14, 14, 512, -1, 3, 3, -1, 2, 1, 1, -1),
# (128, 1024, 14, 14, 512, -1, 1, 1, -1, 1, 0, 1, -1),
# (128, 256, 14, 14, 1024, -1, 1, 1, -1, 1, 0, 1, -1),
# (128, 256, 28, 28, 256, -1, 3, 3, -1, 2, 1, 1, -1),
# (128, 512, 28, 28, 256, -1, 1, 1, -1, 1, 0, 1, -1),
# (128, 128, 28, 28, 512, -1, 1, 1, -1, 1, 0, 1, -1),
# (128, 128, 56, 56, 128, -1, 3, 3, -1, 2, 1, 1, -1),
# (128, 256, 56, 56, 128, -1, 1, 1, -1, 1, 0, 1, -1),
# (128, 64, 56, 56, 256, -1, 1, 1, -1, 1, 0, 1, -1),
# (128, 64, 56, 56, 64, -1, 3, 3, -1, 1, 1, 1, -1),
# (128, 64, 56, 56, 64, -1, 1, 1, -1, 1, 0, 1, -1),
# (128, 256, 56, 56, 64, -1, 1, 1, -1, 1, 0, 1, -1),
# (128, 256, 56, 56, 512, -1, 1, 1, -1, 2, 0, 1, -1),
# (128, 128, 28, 28, 128, -1, 3, 3, -1, 1, 1, 1, -1),
# (128, 512, 28, 28, 128, -1, 1, 1, -1, 1, 0, 1, -1),
# (128, 512, 28, 28, 1024, -1, 1, 1, -1, 2, 0, 1, -1),
# (128, 256, 14, 14, 256, -1, 3, 3, -1, 1, 1, 1, -1),
# (128, 1024, 14, 14, 256, -1, 1, 1, -1, 1, 0, 1, -1),
# (128, 1024, 14, 14, 2048, -1, 1, 1, -1, 2, 0, 1, -1),
# (128, 512, 7, 7, 512, -1, 3, 3, -1, 1, 1, 1, -1),
# (128, 2048, 7, 7, 512, -1, 1, 1, -1, 1, 0, 1, -1),
# (128, 464, 7, 7, 1024, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 320, 64, 64, 320, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 640, 64, 64, 320, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 960, 64, 64, 320, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 640, 64, 64, 640, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 640, 32, 32, 640, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 960, 32, 32, 640, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 1280, 32, 32, 640, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 1920, 32, 32, 640, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 1280, 32, 32, 1280, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 1280, 16, 16, 1280, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 1920, 16, 16, 1280, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 2560, 16, 16, 1280, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 1280, 16, 16, 1280, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 2560, 8, 8, 1280, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 1280, 8, 8, 1280, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 1280, 16, 16, 1280, -1, 3, 3, -1, 2, 1, 1, -1),
# (16, 640, 16, 16, 1280, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 640, 32, 32, 640, -1, 3, 3, -1, 2, 1, 1, -1),
# (16, 320, 32, 32, 640, -1, 1, 1, -1, 1, 0, 1, -1),
# (16, 320, 64, 64, 320, -1, 3, 3, -1, 2, 1, 1, -1),
# (16, 320, 64, 64, 320, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 640, 32, 32, 640, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 320, 32, 32, 640, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 640, 16, 16, 1280, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 1280, 8, 8, 1280, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 2560, 8, 8, 1280, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 2560, 16, 16, 1280, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 1920, 16, 16, 1280, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 1920, 32, 32, 640, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 1280, 32, 32, 640, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 960, 32, 32, 640, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 960, 64, 64, 320, -1, 3, 3, -1, 1, 1, 1, -1),
# (16, 640, 64, 64, 320, -1, 3, 3, -1, 1, 1, 1, -1),
# ]


# benchmark_shapes = [
#     # resnet-18
#     # (N, C, H, W, K, _, R, S, _, stride, padding, dilation, _)
#     # [N, C, H, W, F, -1, K, K, -1, S, P, D,-1],
#     [16, 512, 129, 129, 512, -1, 3, 3, -1, 2, 0, 1,-1],
#     [16, 256, 128, 128, 512, -1, 1, 1, -1, 1, 0, 1,-1],
#     [16, 256, 257, 257, 256, -1, 3, 3, -1, 2, 0, 1,-1],
#     [16, 128, 256, 256, 256, -1, 1, 1, -1, 1, 0, 1,-1],
#     [16, 128, 513, 513, 128, -1, 3, 3, -1, 2, 0, 1,-1],
#     [16, 128, 512, 512, 128, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 256, 256, 256, 256, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 128, 256, 256, 256, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 512, 128, 128, 512, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 256, 128, 128, 512, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 512, 64, 64, 512, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 256, 512, 512, 128, -1, 1, 1, -1, 1, 0, 1,-1],
#     [16, 256, 512, 512, 256, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 512, 256, 256, 256, -1, 1, 1, -1, 1, 0, 1,-1],
#     [16, 512, 256, 256, 512, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 512, 128, 128, 512, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 512, 64, 64, 512, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 256, 256, 256, 256, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 512, 256, 256, 256, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 128, 512, 512, 128, -1, 3, 3, -1, 1, 1, 1,-1],
#     [16, 256, 512, 512, 128, -1, 3, 3, -1, 1, 1, 1,-1],
# ]


benchmark_shapes = [
    # resnet-18
    # (N, C, H, W, K, _, R, S, _, stride, padding, dilation, _)
    # (N, C, H, W, F, -1, K, K, -1, S, P, D, -1),
    (128, 64, 56, 56, 64, -1, 3, 3, -1, 1, 1, 1, -1),
    (128, 64, 56, 56, 64, -1, 1, 1, -1, 1, 0, 1, -1),
    (128, 128, 28, 28, 128, -1, 3, 3, -1, 1, 1, 1, -1),
    (128, 512, 28, 28, 128, -1, 1, 1, -1, 1, 0, 1, -1),
]

example_text = """
    example:
        python conv2d.py --target cuda --batch 256 --enable_cudnn --number 5 --repeats 5 --begin 0 --num 10 --dtype FP16
        python conv2d.py --target llvm --batch 1 --number 10 --repeats 10 --begin 0 --num 5 --dtype INT8
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(benchmark_shapes))), default=0
    )
    parser.add_argument(
        "--num",
        type=int,
        choices=list(range(1, len(benchmark_shapes) + 1)),
        default=len(benchmark_shapes),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["FP16", "FP32", "TF32", "FP64", "BF16", "INT8", "BOOL"],
        default="FP16",
    )
    parser.add_argument("--target", type=str, choices=["cuda", "llvm"], default="cuda")

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False

    RUN_NUMBER = (args.number, args.repeats)
    RUN_CONFIG = (args.batch, args.begin, args.num, args.dtype)

    args = parser.parse_args()
    if args.target == "cuda":
        run_cuda()
    elif args.target == "llvm":
        run_llvm()
