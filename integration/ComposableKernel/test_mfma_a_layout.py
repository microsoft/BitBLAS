import subprocess

layouts = [
    [False, False, False, False, False],
    [False, False, False, False, True],
    [False, False, False, True, False],
    [False, False, False, True, True],
    [False, False, True, False, False],
    [False, False, True, False, True],
    [False, False, True, True, False],
    [False, False, True, True, True],
    [False, True, False, False, False],
    [False, True, False, False, True],
    [False, True, False, True, False],
    [False, True, False, True, True],
    [False, True, True, False, False],
    [False, True, True, False, True],
    [False, True, True, True, False],
    [False, True, True, True, True],

    [True, False, False, False, False],
    [True, False, False, False, True],
    [True, False, False, True, False],
    [True, False, False, True, True],
    [True, False, True, False, False],
    [True, False, True, False, True],
    [True, False, True, True, False],
    [True, False, True, True, True],
    [True, True, False, False, False],
    [True, True, False, False, True],
    [True, True, False, True, False],
    [True, True, False, True, True],
    [True, True, True, False, False],
    [True, True, True, False, True],
    [True, True, True, True, False],
    [True, True, True, True, True],
]

raw_func = '''Fragment makeGemmFragmentACDNA(const int block_m, const int block_n, const int block_k,
                           const int warp_m, const int warp_n) {
  // assume not transposed
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0);
  ICHECK(block_k % 16 == 0);
  auto base_layout = makeGemmFragmentAB16x16CDNA()->Repeat({1, 1}, false, false);
  auto warp_layout = base_layout->Repeat({block_m / warp_m, 1}, true)->Replicate(block_n / warp_n);
  auto block_layout = warp_layout->Repeat({warp_m / 16, block_k / 16}, false, false);
  return block_layout;
}'''
file_path = "/home/aiscuser/lei/BitBLAS/3rdparty/tvm/src/tl/layout/gemm_layouts.cc"

for layout in layouts:
    base_layout_0 = "false" if not layout[0] else "true"
    base_layout_1 = "false" if not layout[1] else "true"
    block_layout_0 = "false" if not layout[2] else "true"
    block_layout_1 = "false" if not layout[3] else "true"
    warp_layout_0 = "false" if not layout[4] else "true"

    log_path = f"base_{base_layout_0}_{base_layout_1}_warp_{warp_layout_0}_block_{block_layout_0}_{block_layout_1}.log"

    new_func = raw_func.replace(
        "auto base_layout = makeGemmFragmentAB16x16CDNA()->Repeat({1, 1}, false, false);",
        f"auto base_layout = makeGemmFragmentAB16x16CDNA()->Repeat({{1, 1}}, {base_layout_0}, {base_layout_1});"
    ).replace(
        "auto warp_layout = base_layout->Repeat({block_m / warp_m, 1}, true)->Replicate(block_n / warp_n);",
        f"auto warp_layout = base_layout->Repeat({{warp_m / 16, block_k / 16}}, {block_layout_0}, {block_layout_0});"
    ).replace(
        "auto block_layout = warp_layout->Repeat({warp_m / 16, block_k / 16}, false, false);",
        f"auto block_layout = warp_layout->Repeat({{block_m / warp_m, 1}}, {warp_layout_0})->Replicate(block_n / warp_n);"
    )
        
    print(new_func)
    with open(file_path, "r") as f:
        content = f.read()
        content = content.replace(raw_func, new_func)
    with open(file_path, "w") as f:
        f.write(content)

    with open(log_path, "w") as log_file:
        # build tvm
        subprocess.run(["make", "-j8"],
                       cwd="/home/aiscuser/lei/BitBLAS/3rdparty/tvm/build",
                       stdout=log_file,
                       stderr=log_file)

        # Execute Test log
        subprocess.run([
            "python",
            "/home/aiscuser/lei/BitBLAS/integration/ComposableKernel/test_mfma_fragement_gemm.py"
        ],
                       cwd="/home/aiscuser/lei/BitBLAS/integration/ComposableKernel",
                       stdout=log_file,
                       stderr=log_file)

    # Recover
    content = content.replace(new_func, raw_func)

    with open(file_path, "w") as f:
        f.write(content)
