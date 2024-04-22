# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from paper_result import (
    llama_providers,
    llama_times_data,
    bloom_providers,
    bloom_times_data
)


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
providers = llama_providers
times_data = llama_times_data

for i, (provider, times) in enumerate(times_data):
    ax_llama.plot(
        providers,
        times,
        label=provider,
        color=colers_sets[i % len(colers_sets)],
        marker="o",
    )

ax_bloom = fig.add_subplot(gs[0, 1])
providers = bloom_providers
times_data = bloom_times_data

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
