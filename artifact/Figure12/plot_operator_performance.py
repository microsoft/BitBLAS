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
providers = ["M0", "M1", "M2", "M3", "M4", "M5"]
times_data = [
    ("cuBLAS", [1.017856, 1.1241472, 31.240396, 0.2287616, 0.2945024, 8.7457794]),
    (
        "cuTLASS-W$_{INT4}$A$_{FP16}$",
        [0.674009323, 1.186704636, 33.67717266, 0.153660774, 0.259065628, 12.6046657],
    ),
    (
        "vLLM-W$_{INT4}$A$_{FP16}$",
        [0.484972, 0.972840786, 123.6705709, 168.7941933, 124.1296554, 168.415212],
    ),
    (
        "Bitter",
        [0.935731232, 1.050994396, 26.89023972, 0.270745605, 0.38573581, 7.485508442],
    ),
    (
        "Bitter-W$_{INT4}$A$_{FP16}$",
        [0.258867204, 0.99830687, 24.94899178, 0.079725713, 0.36928492, 6.955895424],
    ),
    (
        "Bitter-W$_{NF4}$A$_{FP16}$",
        [0.418611199, 1.114526272, 30.12454414, 0.125337601, 0.415465951, 8.341504097],
    ),
    (
        "Bitter-W$_{FP8}$A$_{FP16}$",
        [0.485785574, 0.944679379, 25.32633591, 0.143359989, 0.38356927, 7.078502178],
    ),
    (
        "Bitter-W$_{INT1}$A$_{INT8}$",
        [0.083967999, 0.530721366, 16.49015427, 0.0305152, 0.208530769, 4.851322174],
    ),
    (
        "Bitter-W$_{MXFP8}$A$_{MXFP8}$",
        [0.702719986, 1.678665757, 48.04956055, 0.214783996, 0.617724717, 15.08615303],
    ),
]
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


providers = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
times_data = [
    (
        "Bitter",
        [
            0.175513595,
            0.065536,
            0.143359989,
            0.090112001,
            0.017695315,
            0.00489919,
            0.026812863,
            0.010390706,
        ],
    ),
    (
        "Bitter-W$_{FP8}$A$_{FP16}$",
        [
            0.195993602,
            0.065536,
            0.164659202,
            0.090521596,
            0.01959509,
            0.004884709,
            0.034379646,
            0.013394201,
        ],
    ),
    (
        "Bitter-W$_{MXFP8}$A$_{MXFP8}$",
        [
            0.208281592,
            0.066150397,
            0.200294405,
            0.094617598,
            0.035001777,
            0.006058795,
            0.043367449,
            0.020062251,
        ],
    ),
    (
        "Bitter-W$_{INT4}$A$_{INT4}$",
        [
            0.0978944,
            0.074137598,
            0.082124799,
            0.125337601,
            0.055220462,
            0.009339727,
            0.171313092,
            0.061613843,
        ],
    ),
    (
        "cuDNN",
        [
            0.343142402,
            0.131583999,
            0.248934399,
            0.119091203,
            0.060006401,
            0.058163201,
            0.058572801,
            0.0657408,
        ],
    ),
    (
        "AMOS",
        [
            1.764688,
            0.302502,
            0.857058,
            0.3248,
            0.073223653,
            0.020407328,
            0.106228661,
            0.046321647,
        ],
    ),
    (
        "TensorIR",
        [
            0.2430228,
            0.0932777,
            0.216,
            0.0802,
            0.018251371,
            0.00476218,
            0.026268746,
            0.010746771,
        ],
    ),
]

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
