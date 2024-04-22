import os
from configs import models
import re 
for model_name, model_path in models.items():
    model_name = model_name + "_cutlass"
    run_log_path = os.path.join("compiled_models", model_name, "run.log")
    if not os.path.exists(run_log_path):
        print("no log file for {}".format(model_name))
        continue
    '''
    parse mean time from run.log
        Iteration time 12.396160 ms
        Iteration time 12.387552 ms
        Iteration time 12.334080 ms
        Iteration time 12.508160 ms
        Iteration time 12.329984 ms
        Iteration time 12.326912 ms
        Iteration time 12.365824 ms
        Iteration time 12.329024 ms
        Iteration time 12.323840 ms
        Iteration time 12.383232 ms
        Summary: [min, max, mean] = [12.323840, 12.508160, 12.368476] ms
    '''
    with open(run_log_path, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        pattern = re.compile(r'\[(.*)\]')
        mean_time = pattern.findall(last_line)[0].split(',')[2]
        mean_time = mean_time.split('[')[-1].strip()
        
        print("{}: {}".format(model_name, mean_time))
    