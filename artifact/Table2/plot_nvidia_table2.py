from prettytable import PrettyTable

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--reproduce",
    action="store_true",
    help="reproduce, otherwise use the paper results",
)

args = parser.parse_args()
reproduce = args.reproduce

if not reproduce:
    from paper_result import compilation_cost
else:
    from reproduce_result import compilation_cost

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
