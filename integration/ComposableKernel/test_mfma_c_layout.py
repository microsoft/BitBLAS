import subprocess

layouts = [
    [False, False, False, False],
    [False, False, False, True],
    [False, False, True, False],
    [False, False, True, True],
    [False, True, False, False],
    [False, True, False, True],
    [False, True, True, False],
    [False, True, True, True],
    [True, False, False, False],
    [True, False, False, True],
    [True, False, True, False],
    [True, False, True, True],
    [True, True, False, False],
    [True, True, False, True],
    [True, True, True, False],
    [True, True, True, True],
]

raw_func = '''Fragment makeGemmFragmentCCDNA(const int block_m, const int block_n, const int warp_m, const int warp_n,
                           const int element_size) {
  if (element_size == 64) LOG(FATAL) << "Not supported";
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0) << "warp_m=" << warp_m;
  ICHECK(warp_n % 16 == 0) << "warp_n=" << warp_n;
  auto base_layout = makeGemmFragmentC16x16CDNA()->Repeat({1, 1}, false);
  auto warp_layout = base_layout->Repeat({block_m / warp_m, block_n / warp_n}, false, false);
  auto block_layout = warp_layout->Repeat({warp_m / 16, warp_n / 16}, true, true);
  return block_layout;
}'''
file_path = "/home/aiscuser/lei/BitBLAS/3rdparty/tvm/src/tl/layout/gemm_layouts.cc"

for layout in layouts:
    block_layout_0 = "false" if not layout[0] else "true"
    block_layout_1 = "false" if not layout[1] else "true"
    warp_layout_0 = "false" if not layout[2] else "true"
    warp_layout_1 = "false" if not layout[3] else "true"

    log_path = f"block_{block_layout_0}_{block_layout_1}_warp_{warp_layout_0}_{warp_layout_1}.log"

    # new_func = raw_func.replace(
    #     "base_layout->Repeat({block_m / warp_m, block_n / warp_n}, false, false);",
    #     f"base_layout->Repeat({{block_m / warp_m, block_n / warp_n}}, {block_layout_0}, {block_layout_1});"
    # ).replace(
    #     "warp_layout->Repeat({warp_m / 16, warp_n / 16}, true, true);",
    #     f"warp_layout->Repeat({{warp_m / 16, warp_n / 16}}, {warp_layout_0}, {warp_layout_1});")

    new_func = raw_func.replace(
        "base_layout->Repeat({block_m / warp_m, block_n / warp_n}, false, false);",
        f"base_layout->Repeat({{warp_m / 16, warp_n / 16}}, {block_layout_0}, {block_layout_1});"
    ).replace(
        "warp_layout->Repeat({warp_m / 16, warp_n / 16}, true, true);",
        f"warp_layout->Repeat({{block_m / warp_m, block_n / warp_n}}, {warp_layout_0}, {warp_layout_1});"
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
            "python", "/home/aiscuser/lei/BitBLAS/integration/ComposableKernel/test_block_gemm.py"
        ],
                       cwd="/home/aiscuser/lei/BitBLAS/integration/ComposableKernel",
                       stdout=log_file,
                       stderr=log_file)

    # Recover
    content = content.replace(new_func, raw_func)

    with open(file_path, "w") as f:
        f.write(content)
