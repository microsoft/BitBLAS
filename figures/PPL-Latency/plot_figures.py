import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Data
models = ['LLAMA2-3B', 'LLAMA2-7B', 'LLAMA2-13B', 'LLAMA2-70B']
latencies = {
    'W$_{FP16}$A$_{FP16}$': [5.7171, 8.3704, 17.8364, 82.456],
    'W$_{FP8\_E4M3}$A$_{FP8\_E4M3}$': [4.8071, 5.7144, 11.8364, 48.456],
    'W$_{INT4}$A$_{FP16}$': [3.6371, 3.3464, 8.3564, 27.896],
    'W$_{NF4}$A$_{FP16}$': [4.6511, 5.0104, 11.3164, 43.176],
    'W$_{INT2}$A$_{FP16}$-G64': [3.8711, 3.5704, 8.2364, 27.496],
    'W$_{INT1}$A$_{FP16}$': [3.7151, 3.3464, 8.0764, 25.896],
    'W$_{INT2}$A$_{INT8}$': [3.1951, 2.5144, 6.5164, 17.896],
}

ppls = {
    'W$_{FP16}$A$_{FP16}$': [10.04, 5.47, 4.88, 3.32],
    'W$_{FP8\_E4M3}$A$_{FP8\_E4M3}$': [None, 5.8277, 5.1334, 3.4467],
    'W$_{INT4}$A$_{FP16}$': [None, 6.12, 5.20, 3.67],
    'W$_{NF4}$A$_{FP16}$': [None, 5.87, 5.09, 3.52],
    'W$_{INT2}$A$_{FP16}$-G64': [None, 8.08, 6.78, 5.54],
    'W$_{INT1}$A$_{FP16}$': [None, 9.73, 8.76, None],
    'W$_{INT2}$A$_{INT8}$': [9.91, None, None, None],
}

markers = {
    'W$_{FP16}$A$_{FP16}$': 'o',
    'W$_{FP8\_E4M3}$A$_{FP8\_E4M3}$': 's',
    'W$_{INT4}$A$_{FP16}$': 'D',
    'W$_{NF4}$A$_{FP16}$': '^',
    'W$_{INT2}$A$_{FP16}$-G64': 'v',
    'W$_{INT1}$A$_{FP16}$': '<',
    'W$_{INT2}$A$_{INT8}$': '>',
}

# Using tab10 colormap for higher contrast
colormap = plt.cm.tab10
colors = {
    'LLAMA2-3B': colormap(0.0),
    'LLAMA2-7B': colormap(0.1),
    'LLAMA2-13B': colormap(0.2),
    'LLAMA2-70B': '0.8',
}

# Plotting
plt.figure(figsize=(14, 6))

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

# Custom legend for precision
legend_fontsize = 16
legend_title_fontsize = 18
precision_handles = [plt.scatter([], [], color='black', marker=markers[prec], s=100, facecolors='none') for prec in markers.keys()]
precision_labels = markers.keys()
legend1 = plt.legend(precision_handles, precision_labels, loc="upper center", bbox_to_anchor=(0.85, 1.0), fontsize=legend_fontsize, title='Data Type', shadow=True, fancybox=True, frameon=True, facecolor='white', edgecolor='black', title_fontproperties={'weight':'bold', 'size': legend_title_fontsize})
plt.gca().add_artist(legend1)

# Custom legend for models
model_handles = [plt.scatter([], [], color=colors[mod], s=100) for mod in models]
model_labels = models
legend2 = plt.legend(model_handles, model_labels, loc="upper center", bbox_to_anchor=(0.12, 0.42), fontsize=legend_fontsize, title='Model', shadow=True, fancybox=True, frameon=True, facecolor='white', edgecolor='black', title_fontproperties={'weight':'bold', 'size': legend_title_fontsize})
plt.gca().add_artist(legend2)

# Log scale for x-axis
tick_fontsize = 22
plt.xscale('log', base=2)
# plt.yscale('log', base=2)
# plt.xticks([10, 20, 50, 100], labels=['10', '20', '50', '100'], fontsize=tick_fontsize)
plt.xticks([2, 4, 8, 16, 32, 64, 128], labels=['2', '4', '8', '16', '32', '64', '128'], fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.xlim(left=2)  # Set x-axis limit starting from 1
plt.ylim(bottom=2)  # Set y-axis limit starting from 0
plt.xlabel('Latency (ms)', fontsize=28, labelpad=10,)
plt.ylabel('PPL on WikiText-2(↓)', fontsize=28, labelpad=10)
# plt.title('Latency vs PPL for different precisions and models', fontsize=24, pad=10)
plt.grid(False)

# Save plots
plt.savefig("pdf/ppl_latency.pdf", bbox_inches="tight")
plt.savefig("png/ppl_latency.png", bbox_inches="tight", transparent=False, dpi=255)

print(plt.gca().get_xlim())
