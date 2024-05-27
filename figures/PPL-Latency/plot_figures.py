import matplotlib.pyplot as plt

# Data
models = ['LLAMA2-3B', 'LLAMA2-7B', 'LLAMA2-13B', 'LLAMA2-70B']
latencies = {
    'FP16XFP16': [5.7171, 8.3704, 17.8364, 82.456],
    'FP16XFP8_E4M3-G-1': [4.8071, 5.7144, 13.0364, 48.456],
    'FP16XINT4_G-1 (GPTQ)': [3.6371, 3.3464, 10.3484, 27.896],
    'FP16XNF4_G-1 (NF4)': [4.6511, 5.0104, 12.6204, 43.176],
    'FP16XINT2_G64 (BitDistiller)': [3.8711, 3.5704, 10.1564, 27.496],
    'FP16XINT1 (OneBit)': [3.8451, 3.7624, 10.6364, 30.616],
    'INT8XINT2 (BitNet-b1.58)': [3.1951, 2.5144, 8.7804, 17.896],
}

ppls = {
    'FP16XFP16': [10.04, 5.47, 4.88, 3.32],
    'FP16XFP8_E4M3-G-1': [None, 5.8277, 5.1334, 3.4467],
    'FP16XINT4_G-1 (GPTQ)': [None, 6.12, 5.20, 3.67],
    'FP16XNF4_G-1 (NF4)': [None, 5.87, 5.09, 3.52],
    'FP16XINT2_G64 (BitDistiller)': [None, 8.08, 6.78, 5.54],
    'FP16XINT1 (OneBit)': [None, 9.73, 8.76, None],
    'INT8XINT2 (BitNet-b1.58)': [9.91, None, None, None],
}

markers = {
    'FP16XFP16': 'o',
    'FP16XFP8_E4M3-G-1': 's',
    'FP16XINT4_G-1 (GPTQ)': 'D',
    'FP16XNF4_G-1 (NF4)': '^',
    'FP16XINT2_G64 (BitDistiller)': 'v',
    'FP16XINT1 (OneBit)': '<',
    'INT8XINT2 (BitNet-b1.58)': '>',
}

# Using tab10 colormap for higher contrast
colormap = plt.cm.tab10
colors = {
    'LLAMA2-3B': colormap(0.0),
    'LLAMA2-7B': colormap(0.1),
    'LLAMA2-13B': colormap(0.2),
    'LLAMA2-70B': colormap(0.3),
}

# Plotting
plt.figure(figsize=(14, 8))

for precision in latencies.keys():
    for model in models:
        latency = latencies[precision][models.index(model)]
        ppl = ppls[precision][models.index(model)]
        if latency is not None and ppl is not None:
            color = colors[model]
            marker = markers[precision]
            plt.scatter(latency, ppl, color=color, marker=marker, label=f'{precision} - {model}', s=150, edgecolor='black', linewidth=1.5)

# Connecting points for each model
for model in models:
    model_latencies = [latencies[precision][models.index(model)] for precision in latencies.keys()]
    model_ppls = [ppls[precision][models.index(model)] for precision in ppls.keys()]

    valid_points = [(lat, ppl) for lat, ppl in zip(model_latencies, model_ppls) if lat is not None and ppl is not None]
    valid_points.sort()  # Sort by latency

    # if valid_points:
    #     plt.plot(*zip(*valid_points), color=colors[model])

# Custom legend
handles = []
labels = []
for marker in markers.values():
    handles.append(plt.scatter([], [], color='black', marker=marker, s=100))
labels.extend(markers.keys())
for color in colors.values():
    handles.append(plt.scatter([], [], color=color, s=100))
labels.extend(colors.keys())

plt.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.72, 1.0), ncol=2, fontsize=12, frameon=True, title='Precision - Model', title_fontsize='14', shadow=True, fancybox=True, facecolor='white', edgecolor='black')

# Log scale for x-axis
plt.xscale('log')
plt.xlabel('Log Latency (ms)', fontsize=20)
plt.ylabel('PPL on WikiText2', fontsize=20)
plt.title('Latency vs PPL for different precisions and models', fontsize=24, pad=10)
plt.grid(False)

# Save plots
plt.savefig("pdf/ppl_latency.pdf", bbox_inches="tight")
plt.savefig("png/ppl_latency.png", bbox_inches="tight", transparent=False, dpi=255)
plt.show()