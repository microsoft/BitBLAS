# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker as ticker

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
hatch_patterns = ["-", "+", "x", "\\", "*", "o", "O", "."]

b1s1_providers = [
    "W$_{FP16}$A$_{FP16}$",
    "W$_{INT4}$A$_{FP16}$",
    "W$_{MXFP8}$A$_{MXFP8}$",
    "W$_{INT1}$A$_{INT8}$",
]
llama2_times_data = [
    ("Welder-Roller", [1.206272, 0, 0, 0]),
    ("+Transform", [1.0305, 0.449436135, 1.72799265, 0.26879486]),
    ("+PTX", [1.0305, 0.3437, 1.72799265, 0.1571]),
    ("+Holistic Schedule", [1.0305, 0.3437, 0.8467, 0.1571]),
]

# 创建一个figure实例
fig = plt.figure(figsize=(12, 3.8))
# 获取Torch-Inductor的时间值
_1x_baseline = "+Transform"

# draw for bs1_seq1
_1x_baseline_times = dict(llama2_times_data)[_1x_baseline]

# 计算其他方法相对于Torch-Inductor的加速比
speed_up_data = []
for label, times in llama2_times_data:
    speed_up = [p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)]
    speed_up_data.append((label, speed_up))

# Create an array for x-axis positions
x = np.arange(len(b1s1_providers))


# Set the width of the bars
bar_width = 0.2

max_speedup = np.ceil(max([max(speedup) for _, speedup in speed_up_data]))

# 设置网格布局
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])

# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    print(label, speedup)
    rec = ax0.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=hatch_patterns[i % 8],
        color=colers_sets[i],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            if label == "TVM_TensorIR":
                warning_text = f"Meta Not Support"
            else:
                warning_text = f"Welder Not Support"
            ax0.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.02,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=11,
                rotation=90,
                color="red",
                weight="bold",
            )
ax0.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax0.set_xticklabels(b1s1_providers, fontsize=14)
# set y axis range
ax0.set_ylim(0, 2.3)
b1s4096_providers = [
    "W$_{FP16}$A$_{FP16}$",
    "W$_{INT4}$A$_{FP16}$",
    "W$_{MXFP8}$A$_{MXFP8}$",
    "W$_{INT1}$A$_{INT8}$",
]
b1s4096_llama2_times_data = [
    ("Welder-Roller", [124, 0, 0, 0]),
    ("+Transform", [41.84999636, 37.69587371, 106.5852877, 31.25220761]),
    ("+PTX", [33.7857, 29.94758645, 92.32923136, 24.44975112]),
    ("+Holistic Schedule", [33.7857, 29.94758645, 36.6284164, 24.44975112]),
]
_1x_baseline = "+Transform"

# draw for bs1_seq1
_1x_baseline_times = dict(b1s4096_llama2_times_data)[_1x_baseline]
# 计算其他方法相对于Torch-Inductor的加速比
speed_up_data = []
for label, times in b1s4096_llama2_times_data:
    # if label != _1x_baseline:
    speed_up = [p_i / t if t != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)]
    speed_up_data.append((label, speed_up))

ax1 = fig.add_subplot(gs[0, 1])
# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    rec = ax1.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=hatch_patterns[i % 8],
        color=colers_sets[i],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            if label == "TVM_TensorIR":
                warning_text = f"Meta Not Support"
            else:
                warning_text = f"Welder Not Support"
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.02,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=11,
                rotation=90,
                color="red",
                weight="bold",
            )
ax1.set_xticks(x + len(speed_up_data) * bar_width / 2)
ax1.set_xticklabels(b1s4096_providers, fontsize=14)
# set ylabels font size
ax1.set_yticklabels(ax1.get_yticks(), fontsize=10)

# 调整图例位置和大小
legend_fontsize = 16
handles, labels = ax0.get_legend_handles_labels()

order = ["Welder-Roller", "+Transform", "+PTX", "+Holistic Schedule"]

# Create new lists in the desired order
ordered_handles = []
ordered_labels = []

for label in order:
    index = labels.index(label)
    ordered_handles.append(handles[index])
    ordered_labels.append(labels[index])
# set ax0's y font size

plt.legend(
    ordered_handles,
    ordered_labels,
    ncol=len(ordered_handles),
    bbox_to_anchor=(0.8, 1.28),
    fontsize=legend_fontsize,
    frameon=False,
)

xlabel_fontsize = 15
ax0.set_xlabel("BS1 SEQ1", fontsize=xlabel_fontsize, labelpad=10)
ax1.set_xlabel("BS1 SEQ4096", fontsize=xlabel_fontsize, labelpad=10)
ax0.set_ylabel("Normalized Speedup", fontsize=16, labelpad=10)

ax1.set_ylim(0, max_speedup)
# Generate a list of labels based on the new y-axis range
# y_ticks = np.arange(0, max_speedup * 1.1, 1.0)  # You can define the step based on your preference
# ax0.set_yticks(y_ticks)  # This sets the y-ticks to be at the locations you specify
# ax0.set_yticklabels([str(tick) for tick in y_ticks], fontsize=10)  # This sets the y-tick labels with the desired fontsize

# 调整布局以避免图例被遮挡
# plt.subplots_adjust(top=0.75, bottom=-0.15, left=0.15)
plt.subplots_adjust(top=0.75, bottom=0.2, left=0.13, right=0.94)

# 保存图形
plt.savefig(
    "pdf/optimization_breakdown.pdf",
    bbox_inches="tight",
)
plt.savefig(
    "png/optimization_breakdown.png",
    bbox_inches="tight",
    transparent=False,
    dpi=255,
)
