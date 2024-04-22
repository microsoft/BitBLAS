# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker as ticker
from paper_result.data_a6000 import *

colers_sets = [
    # nilu
    (20 / 255, 54 / 255, 95 / 255),
    (118 / 255, 162 / 255, 185 / 255),
    (191 / 255, 217 / 255, 229 / 255),
    (214 / 255, 79 / 255, 56 / 255),
    (112 / 255, 89 / 255, 146 / 255),
    # dori
    (169 / 255, 115 / 255, 153 / 255),
    (248 / 255, 242 / 255, 236 / 255),
    (214 / 255, 130 / 255, 148 / 255),
    (243 / 255, 191 / 255, 202 / 255),
    # xiao
    (49 / 255, 102 / 255, 88 / 255),
    (94 / 255, 166 / 255, 156 / 255),
    (194 / 255, 207 / 255, 162 / 255),
    (164 / 255, 121 / 255, 158 / 255),
    # coller
    (124 / 255, 134 / 255, 65 / 255),
    (185 / 255, 198 / 255, 122 / 255),
    (248 / 255, 231 / 255, 210 / 255),
    (182 / 255, 110 / 255, 151 / 255),
]

# 创建一个figure实例
fig = plt.figure(figsize=(16, 4))

# 获取Torch-Inductor的时间值
_1x_baseline = "Welder"

# 设置网格布局
gs = gridspec.GridSpec(2, 12, figure=fig, height_ratios=[1, 1], wspace=0.3, hspace=1.2)

# 第一行的两个大图（需要截断的Y轴）
# llama2
# 这里添加用于绘制llama2图表的代码
########################## Llama Plot#######################################
# Data
providers = llama2_providers
times_data = llama2_times_data

hatch_patterns = ["-", "+", "x", "\\", "*", "o", "O", "."]

legend_items = {}

llm_legands = []
other_legands = []


def get_legend_item(label):
    if label not in legend_items:
        idx = len(legend_items)
        legend_items[label] = (
            colers_sets[idx % len(colers_sets)],
            hatch_patterns[idx % len(hatch_patterns)],
        )
    return legend_items[label]


providers_bs1_seq1 = ["BS1 SEQ1"]
providers_bs32_seq1 = ["BS32 SEQ1"]
providers_bs1_seq4096 = ["BS1 SEQ4096"]

times_data_bs1_seq1 = [(label, [times[0]]) for label, times in times_data]
times_data_bs32_seq1 = [(label, [times[1]]) for label, times in times_data]
times_data_bs1_seq4096 = [(label, [times[2]]) for label, times in times_data]
providers = providers_bs1_seq1
times_data = times_data_bs1_seq1

gs_llama2_bs1_seq1 = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs[0, 0:2], hspace=0.25
)
ax1a = fig.add_subplot(gs_llama2_bs1_seq1[0])
ax1b = fig.add_subplot(gs_llama2_bs1_seq1[1:])

# draw for bs1_seq1
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于Torch-Inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))


# Create an array for x-axis positions
x = np.arange(len(providers))


# Set the width of the bars
bar_width = 0.07

max_speedup = np.ceil(max([max(speedup) for _, speedup in speed_up_data]))

# 设置两个Y轴的范围
ax1a.set_ylim(10, max_speedup)  # 上面的图为10到最大值
ax1b.set_ylim(0, 5)  # 下面的图为0到5


# Draw cublas as a horizontal dashed line
ax1a.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)
ax1b.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)
ax1a.spines["bottom"].set_visible(False)
ax1b.spines["top"].set_visible(False)
ax1a.set_xticklabels([])
ax1a.set_xticks([])
# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    if label not in llm_legands:
        llm_legands.append(label)
    rec = ax1a.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    ax1b.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

    for rect in rec:
        height = rect.get_height()
        if height == 0:
            if label == "TensorIR":
                warning_text = f"TIR Not Support"
            else:
                warning_text = f"{label} Not Support"
            ax1a.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
                color="red",
                weight="bold",
            )

d = 0.01  # 斜线的长度
kwargs = dict(transform=ax1a.transAxes, color="k", clip_on=False)
ax1a.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
ax1a.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal


kwargs.update(transform=ax1b.transAxes)  # switch to the bottom axes
ax1b.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax1b.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax1b.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax1b.set_xticklabels(providers)
ax1b.grid(False)

providers = providers_bs32_seq1
times_data = times_data_bs32_seq1
# draw for bs1_seq1
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于Torch-Inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))

ax1_1 = fig.add_subplot(gs[0, 2:4])
x = np.arange(len(providers))


# Set the width of the bars
bar_width = 0.09

# Draw cublas as a horizontal dashed line
ax1_1.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)


# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    if label not in llm_legands:
        llm_legands.append(label)
    rec = ax1_1.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            if label == "TensorIR":
                warning_text = f"TIR Not Support"
            else:
                warning_text = f"{label} Not Support"
            ax1_1.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
                color="red",
                weight="bold",
            )

ax1_1.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax1_1.set_xticklabels(providers)

providers = providers_bs1_seq4096
times_data = times_data_bs1_seq4096
# draw for bs1_seq1
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于Torch-Inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))

ax1_2 = fig.add_subplot(gs[0, 4:6])
x = np.arange(len(providers))


# Set the width of the bars
bar_width = 0.09

# Draw cublas as a horizontal dashed line
ax1_2.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)


# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    if label not in llm_legands:
        llm_legands.append(label)
    rec = ax1_2.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            if label == "TensorIR":
                warning_text = f"TIR Not Support"
            else:
                warning_text = f"{label} Not Support"
            ax1_2.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
                color="red",
                weight="bold",
            )

ax1_2.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax1_2.set_xticklabels(providers)


# bloom
# 这里添加用于绘制bloom图表的代码
# Data
providers = bloom_providers
times_data = bloom_times_data
providers_bs1_seq1 = ["BS1 SEQ1"]
providers_bs32_seq1 = ["BS32 SEQ1"]
providers_bs1_seq4096 = ["BS1 SEQ4096"]

times_data_bs1_seq1 = [(label, [times[0]]) for label, times in times_data]
times_data_bs32_seq1 = [(label, [times[1]]) for label, times in times_data]
times_data_bs1_seq4096 = [(label, [times[2]]) for label, times in times_data]
providers = providers_bs1_seq1
times_data = times_data_bs1_seq1
gs_bloom = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 6:8], hspace=0.25)
ax2a = fig.add_subplot(gs_bloom[0])
ax2b = fig.add_subplot(gs_bloom[1:])

# 获取Torch-Inductor的时间值
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于Torch-Inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))

# Create an array for x-axis positions
x = np.arange(len(providers))
# Set the width of the bars
bar_width = 0.07

max_speedup = np.ceil(max([max(speedup) for _, speedup in speed_up_data]))

# 设置两个Y轴的范围
ax2a.set_ylim(13, max_speedup * 1.05)  # 上面的图为10到最大值
ax2b.set_ylim(0, 6)  # 下面的图为0到5


# Draw cublas as a horizontal dashed line
ax2a.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)
ax2b.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)
ax2a.spines["bottom"].set_visible(False)
ax2b.spines["top"].set_visible(False)
ax2a.set_xticklabels([])
ax2a.set_xticks([])

# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    if label not in llm_legands:
        llm_legands.append(label)
    rec = ax2a.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    ax2b.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

    for rect in rec:
        height = rect.get_height()
        if height == 0:
            if label == "TensorIR":
                warning_text = f"TIR Not Support"
            else:
                warning_text = f"{label} Not Support"
            ax2a.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
                color="red",
                weight="bold",
            )

d = 0.01  # 斜线的长度
kwargs = dict(transform=ax2a.transAxes, color="k", clip_on=False)
ax2a.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
ax2a.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal


kwargs.update(transform=ax2b.transAxes)  # switch to the bottom axes
ax2b.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2b.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax2b.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax2b.set_xticklabels(providers)
ax2a.grid(False)
ax2b.grid(False)


providers = providers_bs32_seq1
times_data = times_data_bs32_seq1
# draw for bs1_seq1
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于Torch-Inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))

ax2_1 = fig.add_subplot(gs[0, 8:10])
x = np.arange(len(providers))


# Set the width of the bars
bar_width = 0.09

# Draw cublas as a horizontal dashed line
ax2_1.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)


# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    if label not in llm_legands:
        llm_legands.append(label)
    rec = ax2_1.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            if label == "TensorIR":
                warning_text = f"TIR Not Support"
            else:
                warning_text = f"{label} Not Support"
            ax2_1.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
                color="red",
                weight="bold",
            )

ax2_1.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax2_1.set_xticklabels(providers)

providers = providers_bs1_seq4096
times_data = times_data_bs1_seq4096
# draw for bs1_seq1
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于Torch-Inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))

ax2_2 = fig.add_subplot(gs[0, 10:12])
x = np.arange(len(providers))


# Set the width of the bars
bar_width = 0.09

# Draw cublas as a horizontal dashed line
ax2_2.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)


# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    if label not in llm_legands:
        llm_legands.append(label)
    rec = ax2_2.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            if label == "TensorIR":
                warning_text = f"TIR Not Support"
            else:
                warning_text = f"{label} Not Support"
            ax2_2.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
                color="red",
                weight="bold",
            )

ax2_2.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax2_2.set_xticklabels(providers)

# 第二行的四个小图
ax3 = fig.add_subplot(gs[1, 0:3])  # ResNet
# 加载和显示ResNet图表
# Data
providers = resnet_providers
times_data = resnet_times_data
# 获取Torch-Inductor的时间值
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于Torch-Inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))

# Create an array for x-axis positions
x = np.arange(len(providers))


# Set the width of the bars
bar_width = 0.09

# Draw cublas as a horizontal dashed line
ax3.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)


# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    if label not in other_legands:
        other_legands.append(label)
    ax3.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

# X-axis and labels
ax3.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax3.set_xticklabels(providers)

ax3.grid(False)

ax4 = fig.add_subplot(gs[1, 3:6])  # ShuffleNet
# 加载和显示ShuffleNet图表
# Data
providers = shufflenet_providers
times_data = shufflenet_times_data
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于Torch-Inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))

# Create an array for x-axis positions
x = np.arange(len(providers))


# Set the width of the bars
bar_width = 0.09

# Draw cublas as a horizontal dashed line
ax4.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)


# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    if label not in other_legands:
        other_legands.append(label)
    ax4.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

# X-axis and labels
# ax4.set_xlabel('ShuffleNet')
# ax4.set_ylabel('Speedup vs. Welder')
ax4.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax4.set_xticklabels(providers)

ax4.grid(False)

ax5 = fig.add_subplot(gs[1, 6:9])  # conformer
# 加载和显示conformer图表
# Data
providers = conformer_providers
times_data = conformer_times_data

_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于Torch-Inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))


# Create an array for x-axis positions
x = np.arange(len(providers))


# Set the width of the bars
bar_width = 0.09


# Draw cublas as a horizontal dashed line
ax5.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)


# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    if label not in other_legands:
        other_legands.append(label)
    rec = ax5.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            if label == "TensorIR":
                warning_text = f"TIR Not Support"
            else:
                warning_text = f"{label} Not Support"
            ax5.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
                color="red",
                weight="bold",
            )

# X-axis and labels
# ax5.set_xlabel('ShuffleNet')
# ax5.set_ylabel('Speedup vs. Welder')
ax5.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax5.set_xticklabels(providers)

ax5.grid(False)


ax6 = fig.add_subplot(gs[1, 9:12])  # ViT
# 加载和显示ViT图表
# Data
providers = vit_providers
times_data = vit_times_data

_1x_baseline_times = dict(times_data)[_1x_baseline]
# 计算其他方法相对于Torch-Inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))


# Draw cublas as a horizontal dashed line
ax6.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)


# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    if label not in other_legands:
        other_legands.append(label)
    rec = ax6.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            if label == "TensorIR":
                warning_text = f"TIR Not Support"
            else:
                warning_text = f"{label} Not Support"
            ax6.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
                color="red",
                weight="bold",
            )

# X-axis and labels
# ax6.set_xlabel('ShuffleNet')
# ax6.set_ylabel('Speedup vs. Welder')
ax6.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax6.set_xticklabels(providers)

ax6.grid(False)


# 假设ax1a, ax1b, ax2a, ax2b, ax3, ax4, ax5, ax6是您的子图对象
# 在每个子图的外部底部添加脚注

legand_font = 10
ax1b.text(
    0.5,
    -0.43,
    "(a) LLAMA",
    transform=ax1_1.transAxes,
    fontsize=legand_font,
    ha="center",
)
ax2b.text(
    0.5,
    -0.43,
    "(b) BLOOM",
    transform=ax2_1.transAxes,
    fontsize=legand_font,
    ha="center",
)
ax3.text(
    0.5, -0.3, "(c) ResNet", transform=ax3.transAxes, fontsize=legand_font, ha="center"
)
ax4.text(
    0.5,
    -0.3,
    "(d) ShuffleNet",
    transform=ax4.transAxes,
    fontsize=legand_font,
    ha="center",
)
ax5.text(
    0.5,
    -0.3,
    "(e) Conformer",
    transform=ax5.transAxes,
    fontsize=legand_font,
    ha="center",
)
ax6.text(
    0.5, -0.3, "(f) ViT", transform=ax6.transAxes, fontsize=legand_font, ha="center"
)

y_size = 8
ax1a.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax1b.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax1_1.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax1_2.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax2a.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax2b.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax2_1.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax2_2.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax3.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax4.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax5.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax6.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小

axes = [ax1a, ax1b, ax1_1, ax1_2, ax2a, ax2b, ax2_1, ax2_2, ax3, ax4, ax5, ax6]
for ax in axes:
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# set the y-label for the gs_llama2_bs1_seq1
# 获取ax1a的位置信息
# 计算ax1a和ax1b的垂直位置
# 获取ax3的位置信息
pos_ax3 = ax3.get_position()
left_ax3 = pos_ax3.x0 - 0.028

# 计算ax1a和ax1b的垂直位置
y0_ax1a = ax1a.get_position().y0
y1_ax1b = ax1b.get_position().y1

# 计算中点位置
middle_y = (y0_ax1a + y1_ax1b) / 2 - 0.05

# 添加文本作为Y轴标签，并确保与ax3的Y轴标签对齐
fig.text(
    left_ax3,
    middle_y,
    "Speedup vs. Welder",
    va="center",
    rotation="vertical",
    fontsize=10,
)

# 设置ax3的Y轴标签
ax3.set_ylabel("Speedup vs. Welder", fontsize=10, labelpad=10)


# get handles from ax1a and ax1b and get labels from ax1_1 and ax1_2
handles1, labels1 = ax1_2.get_legend_handles_labels()

# 调整图例位置和大小
legend_fontsize = 10
llm_axes = [ax1a, ax1b, ax1_1, ax1_2, ax2a, ax2b, ax2_1, ax2_2]
handles_llm = []
labels_llm = []
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in labels_llm and label in llm_legands:
            handles_llm.append(handle)
            labels_llm.append(label)
        else:
            pass


# 为上面六个图添加图例
fig.legend(
    handles_llm,
    labels_llm,
    loc="upper center",
    ncol=len(labels_llm) // 2 + 1,
    fontsize=legend_fontsize,
    frameon=True,
)

# 获取图例的handles和labels
other_axes = [ax3, ax4, ax5, ax6]
handles_other = []
labels_other = []
handles_bitter = []
labels_bitter = []
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in (labels_other + labels_bitter) and label in other_legands:
            if "Bitter" in label:
                handles_bitter.append(handle)
                labels_bitter.append(label)
            else:
                handles_other.append(handle)
                labels_other.append(label)
        else:
            pass
handles_other.extend(handles_bitter)
labels_other.extend(labels_bitter)
# 调整图例位置和大小
legend_fontsize = 10

# 将图例放置在图表中间
fig.legend(
    handles_other,
    labels_other,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.52),
    ncol=len(labels_other) // 2,
    fontsize=legend_fontsize,
    frameon=True,
)

# 调整布局以避免图例被遮挡
plt.subplots_adjust(top=0.85, bottom=0.15)


# 保存图形
plt.savefig(
    "pdf/end2end_a6000.pdf",
    bbox_inches="tight",
)
plt.savefig(
    "png/end2end_a6000.png",
    bbox_inches="tight",
    transparent=False,
    dpi=255,
)
