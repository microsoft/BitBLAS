# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--reproduce", action="store_true", help="reproduce, otherwise use the paper results", default=False)

args = parser.parse_args()

reproduce = args.reproduce

if not reproduce:
    from paper_result import (
        matmul_providers,
        matmul_times_data,
        conv_providers,
        conv_times_data
    )
else:
    from reproduce_result import (
        matmul_providers,
        matmul_times_data,
        conv_providers,
        conv_times_data
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
    (248 / 255, 242 / 255, 236 / 255),
    (214 / 255, 130 / 255, 148 / 255),
    (243 / 255, 191 / 255, 202 / 255),
    # (41/ 255, 31/ 255, 39/ 255),
    # coller
    # (72/ 255, 76/ 255, 35/ 255),
    (124 / 255, 134 / 255, 65 / 255),
    (185 / 255, 198 / 255, 122 / 255),
    (248 / 255, 231 / 255, 210 / 255),
    (182 / 255, 110 / 255, 151 / 255),
]
hatch_patterns = ["-", "+", "x", "\\", "*", "o", "O", "."]


# paper result
providers = matmul_providers
times_data = matmul_times_data

# 创建一个figure实例
fig = plt.figure(figsize=(6, 3))

# 获取Torch-Inductor的时间值
_1x_baseline = "Bitter"
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


# 设置网格布局
gs = gridspec.GridSpec(
    3, 1, figure=fig, height_ratios=[0.5, 1, 1.5], wspace=0.1, hspace=0.4
)


ax1_1_2 = fig.add_subplot(gs[0, 0])
ax1_1 = fig.add_subplot(gs[1, 0])
# 设置两个Y轴的范围
ax1_1_2.set_ylim(8, 14)  # 上面的图为10到最大值
ax1_1.set_ylim(0, 4)  # 下面的图为0到5

# Draw cublas as a horizontal dashed line
ax1_1_2.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)
ax1_1.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)
ax1_1_2.spines["bottom"].set_visible(False)
ax1_1.spines["top"].set_visible(False)
ax1_1_2.set_xticklabels([])
ax1_1_2.set_xticks([])

times_data = times_data
providers = providers
# 获取pytorch_inductor的时间值
_1x_baseline = "Bitter"
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于pytorch_inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))
# Plotting
# fig, ax = plt.subplots(figsize=(6, 2))

max_speedup = np.ceil(max([max(speedup) for _, speedup in speed_up_data]))

print(speed_up_data)
# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.07

# Draw cublas as a horizontal dashed line
ax1_1.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)


# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    if label not in llm_legands:
        llm_legands.append(label)

    rec = ax1_1_2.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
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
                if "vLLM" in label:
                    warning_text = "vLLM Not Support"
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


d = 0.01  # 斜线的长度
kwargs = dict(transform=ax1_1_2.transAxes, color="k", clip_on=False)
ax1_1_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
ax1_1_2.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal


kwargs.update(transform=ax1_1.transAxes)  # switch to the bottom axes
ax1_1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax1_1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax1_1.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax1_1.set_xticklabels(providers)
ax1_1.grid(False)

ax1_1.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax1_1.set_xticklabels(providers)
# ax1_1.set_ylabel('Speedup Vs. Bitter', fontsize=12, labelpad=10)


providers = conv_providers
times_data = conv_times_data
ax2_1 = fig.add_subplot(gs[2, 0])
times_data = times_data
providers = providers
# 获取pytorch_inductor的时间值
_1x_baseline = "Bitter"
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于pytorch_inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))
# Plotting
# fig, ax = plt.subplots(figsize=(6, 2))

max_speedup = np.ceil(max([max(speedup) for _, speedup in speed_up_data]))

print(speed_up_data)
# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.07

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
                if "vLLM" in label:
                    warning_text = "vLLM Not Support"
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

legend_fontsize = 6

handles_other = []
labels_other = []
handles_bitter = []
labels_bitter = []
for ax in [ax1_1_2, ax1_1, ax2_1]:
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in (labels_other + labels_bitter):
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
print(handles_other)
# 调整图例位置和大小
legend_fontsize = 7
fig.legend(
    handles_other,
    labels_other,
    loc="upper left",
    bbox_to_anchor=(0.76, 0.89),
    ncol=1,
    fontsize=legend_fontsize,
    frameon=True,
)

fig.text(
    0.07,
    0.45,
    "Speedup Vs. Bitter-W$_{FP16}$A$_{FP16}$",
    fontsize=9,
    rotation=90,
    va="center",
    ha="center",
)
plt.subplots_adjust(top=0.9, bottom=0.15, right=0.75)

# 保存图形
plt.savefig(
    "pdf/operator_performance_a100.pdf",
    bbox_inches="tight",
)
plt.savefig(
    "png/operator_performance_a100.png",
    bbox_inches="tight",
    transparent=False,
    dpi=255,
)
