# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker as ticker
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--reproduce", action="store_true", help="reproduce, otherwise use the paper results", default=False)

args = parser.parse_args()

reproduce = args.reproduce

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

if not reproduce:
    from paper_result import (
        b1s1_providers,
        b1s1_times_data,
        b1s4096_providers,
        b1s4096_times_data,
    )
else:
    from reproduce_result import (
        b1s1_providers,
        b1s1_times_data,
        b1s4096_providers,
        b1s4096_times_data,
    )

# 创建一个figure实例
fig = plt.figure(figsize=(8, 4))

# 获取Torch-Inductor的时间值
_1x_baseline = "Welder"
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
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], wspace=0.1, hspace=0.1)

ax1_1 = fig.add_subplot(gs[0, 0])
times_data = b1s1_times_data
providers = b1s1_providers
# 获取pytorch_inductor的时间值
_1x_baseline = "Ladder-W$_{FP16}$A$_{FP16}$"
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

ax1_1.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax1_1.set_xticklabels(providers)
# ax1_1.set_ylabel('Speedup Vs. Ladder', fontsize=12, labelpad=10)


ax1_2 = fig.add_subplot(gs[0, 1])
times_data = b1s4096_times_data
providers = b1s4096_providers
# 获取pytorch_inductor的时间值
_1x_baseline = "Ladder-W$_{FP16}$A$_{FP16}$"
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
                if "vLLM" in label:
                    warning_text = "vLLM Not Support"
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

ax2_1 = fig.add_subplot(gs[1, 0])

if not reproduce:
    from paper_result import b1s1_matmul_providers as providers, b1s1_matmul_times_data as times_data
    
else:
    from reproduce_result import b1s1_matmul_providers as providers, b1s1_matmul_times_data as times_data

# 获取pytorch_inductor的时间值
_1x_baseline = "Ladder-W$_{FP16}$A$_{FP16}$"
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于pytorch_inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))

max_speedup = np.ceil(max([max(speedup) for _, speedup in speed_up_data]))


ax2_1.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)

x = np.arange(len(providers))

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
# ax2_1.set_ylabel('Speedup Vs. Ladder', fontsize=12, labelpad=10, )

ax2_2 = fig.add_subplot(gs[1, 1])

if not reproduce:
    from paper_result import b1s4096_matmul_providers as providers, b1s4096_matmul_times_data as times_data
    
else:
    from reproduce_result import b1s4096_matmul_providers as providers, b1s4096_matmul_times_data as times_data

# 获取pytorch_inductor的时间值
_1x_baseline = "Ladder-W$_{FP16}$A$_{FP16}$"
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对于pytorch_inductor的加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up = [
            p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        speed_up_data.append((label, speed_up))

max_speedup = np.ceil(max([max(speedup) for _, speedup in speed_up_data]))


ax2_2.axhline(y=1, color="black", linestyle="dashed", label=_1x_baseline)

x = np.arange(len(providers))

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
                if "vLLM" in label:
                    warning_text = "vLLM Not Support"
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

handles, labels = ax1_1.get_legend_handles_labels()
legend_fontsize = 9

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=len(labels) // 3,
    fontsize=legend_fontsize,
    frameon=True,
)

legand_font = 10
# text speedup vs Ladder-fp16, transpose and let it at the left
fig.text(
    0.07,
    0.45,
    "Speedup Vs. Ladder-W$_{FP16}$A$_{FP16}$",
    fontsize=14,
    rotation=90,
    va="center",
    ha="center",
)
fig.text(
    0.5,
    -1.5,
    "(a) BS1 SEQ1",
    transform=ax1_1.transAxes,
    fontsize=legand_font,
    ha="center",
    va="bottom",
)
fig.text(
    1.5,
    -1.5,
    "(b) BS1 SEQ4096",
    transform=ax1_1.transAxes,
    fontsize=legand_font,
    ha="center",
    va="bottom",
)
# fig.text
# 调整布局以避免图例被遮挡
plt.subplots_adjust(top=0.75, bottom=0.15)

# 保存图形
plt.savefig(
    "pdf/figure14_different_bits.pdf",
    bbox_inches="tight",
)
plt.savefig(
    "png/figure14_different_bits.png",
    bbox_inches="tight",
    transparent=False,
    dpi=255,
)
