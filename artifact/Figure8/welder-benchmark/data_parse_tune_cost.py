# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from configs import models
import re 
for model_name, model_path in models.items():
    model_name = model_name + "_cutlass"
    run_log_path = os.path.join("compiled_models", model_name, "tune_time_cost.txt")
    if not os.path.exists(run_log_path):
        print("no log file for {}".format(model_name))
        continue
    """
        Compiler tuning time: 536 seconds
    """
    with open(run_log_path, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        pattern = re.compile(r'Compiler tuning time: (.*) seconds')
        mean_time = pattern.findall(last_line)
        mean_time = mean_time[0]
        print("{}: {}".format(model_name, mean_time))
