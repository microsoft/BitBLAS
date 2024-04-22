import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

colers_sets = [
    # nilu
    (20 / 255, 54 / 255, 95 / 255),
    (118 / 255, 162 / 255, 185 / 255),
    (191 / 255, 217 / 255, 229 / 255),
    (214 / 255, 79 / 255, 56 / 255),
    (112 / 255, 89 / 255, 146 / 255),
    # dori
    (169 / 255, 115 / 255, 153 / 255),
    # (248/ 255, 242/ 255, 236/ 255),
    (214 / 255, 130 / 255, 148 / 255),
    (243 / 255, 191 / 255, 202 / 255),
    (41 / 255, 31 / 255, 39 / 255),
    # coller
    (72 / 255, 76 / 255, 35 / 255),
    (124 / 255, 134 / 255, 65 / 255),
    (185 / 255, 198 / 255, 122 / 255),
    (248 / 255, 231 / 255, 210 / 255),
    (182 / 255, 110 / 255, 151 / 255),
]
hatch_patterns = ["-", "+", "x", "\\", "*", "o", "O", "."]

# 创建一个figure实例
fig = plt.figure(figsize=(9, 3))
# 设置网格布局
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)


ax_llama = fig.add_subplot(gs[0, 0])
providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
times_data = [
    ("PyTorch-Inductor", [2660, 2642, 6754]),
    ("ONNXRuntime", [2748, 2780, 16206]),
    ("TensorRT", [5140, 5148, 6260]),
    ("vLLM", [4866, 4868, 4866]),
    ("vLLM-W$_{INT4}$A$_{FP16}$", [1072, 1072, 6400]),
    ("Welder", [2076, 2084, 6626]),
    ("Bitter", [2064, 2070, 6580]),
    ("Bitter-W$_{INT4}$A$_{FP16}$", [840, 846, 5356]),
    ("Bitter-W$_{NF4}$A$_{FP16}$", [852, 853, 5364]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [1248, 1254, 5764]),
    ("Bitter-W$_{MXFP8}$A$_{MXFP8}$", [1248, 1254, 5764]),
    ("Bitter-W$_{INT1}$A$_{INT8}$", [534, 540, 5050]),
]

for i, (provider, times) in enumerate(times_data):
    ax_llama.plot(
        providers,
        times,
        label=provider,
        color=colers_sets[i % len(colers_sets)],
        marker="o",
    )

ax_bloom = fig.add_subplot(gs[0, 1])
providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
times_data = [
    ("PyTorch-Inductor", [12088, 12072, 15674]),
    ("ONNXRuntime", [7356, 6844, 64718]),
    ("TensorRT", [5771, 5783, 21292]),
    ("vLLM", [30512, 30516, 30512]),
    ("vLLM-W$_{INT4}$A$_{FP16}$", [22608, 22612, 22608]),
    ("Welder", [5148, 5160, 20046]),
    ("Bitter", [5136, 5156, 20592]),
    ("Bitter-W$_{INT4}$A$_{FP16}$", [3372, 3392, 18828]),
    ("Bitter-W$_{NF4}$A$_{FP16}$", [3382, 3384, 18844]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [3960, 3980, 19416]),
    ("Bitter-W$_{MXFP8}$A$_{MXFP8}$", [3960, 3980, 19416]),
    ("Bitter-W$_{INT1}$A$_{INT8}$", [2931, 2951, 18387]),
]

for i, (provider, times) in enumerate(times_data):
    ax_bloom.plot(
        providers,
        times,
        label=provider,
        color=colers_sets[i % len(colers_sets)],
        marker="o",
    )


legend_fontsize = 8
# set y scale
ax_llama.set_yscale("log")
ax_llama.set_xlabel("LLAMA", fontsize=15, labelpad=10)
ax_bloom.set_yscale("log")
ax_bloom.set_xlabel("BLOOM", fontsize=15, labelpad=10)
# set the font size
ax_llama.set_ylabel("Memory Usage (MB)", fontsize=15, labelpad=10)

handles, labels = ax_llama.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(labels) // 2,
    fontsize=legend_fontsize,
    frameon=True,
    bbox_to_anchor=(0.5, 1.01),
)

# 调整布局以避免图例被遮挡
plt.subplots_adjust(top=0.8, bottom=0.2)

plt.grid(False)

# 保存图形
plt.savefig(
    "pdf/memory_usage_a100.pdf",
    bbox_inches="tight",
)
plt.savefig(
    "png/memory_usage_a100.png",
    bbox_inches="tight",
    transparent=False,
    dpi=255,
)
