from prettytable import PrettyTable


compilation_cost = {
    "ResNet(1)": {
        "AMOS": "3852",
        "TensorIR": "156",
        "Welder": "11",
        "LADDER": "31"
    },
    "ResNet(128)": {
        "AMOS": "3328",
        "TensorIR": "128",
        "Welder": "13",
        "LADDER": "17"
    },
    "ShuffleNet(1)": {
        "AMOS": "2191",
        "TensorIR": "836",
        "Welder": "18",
        "LADDER": "44"
    },
    "ShuffleNet(128)": {
        "AMOS": "3121",
        "TensorIR": "400",
        "Welder": "12",
        "LADDER": "29"
    }
}

# TODO(Append LOG Parser to  update the table for this script)

# initialize the figures
table = PrettyTable()
table.title = f"Compilation  time (in minutes)  comparison of end-to-end models on NVIDIA A100 GPU"
table.field_names = ["Model(BS)", "ResNet(1)", "ResNet(128)", "ShuffleNet(1)", "ShuffleNet(128)"]

# collect the transposed data
transposed_data = {key: [] for key in table.field_names[1:]}  # 初始化库对应的列表
libraries = ["AMOS", "TensorIR", "Welder", "LADDER"]

for lib in libraries:
    row = [lib]
    for precision in table.field_names[1:]:
        row.append(compilation_cost[precision].get(lib, 'x'))
    table.add_row(row)

print(table)
