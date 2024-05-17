import os
import re
from conv2d_data_config import conv_infos

log_dir = './logs'


def parse_algo_and_runtime(log_path):
    pattern = r'\d+\.\d+|\d+' # regex pattern to recognize float data
    algo = 0
    meantime = 0.0
    with open(log_path) as f:
        lines = f.readlines()
        for line in lines:
            if "Select algo" in line:
                algo = re.findall(pattern, line)
                if len(algo):
                    algo = algo[0]
            if "raw_data" in line:
                meantime = re.findall(pattern, line)
                if len(meantime):
                    meantime = meantime[0]
    return algo, meantime


for conv_info in conv_infos:
    f32_log_name = 'cudnn_shape_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_performance.float32.log'.format(conv_info['N'],conv_info['C'],conv_info['H'],conv_info['W'],conv_info['F'],conv_info['K'],conv_info['S'],conv_info['D'],conv_info['P'], conv_info['HO'], conv_info['WO'])
    f32_log_path = os.path.join(log_dir, f32_log_name)
    f16_log_name = 'cudnn_shape_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_performance.float16.log'.format(conv_info['N'],conv_info['C'],conv_info['H'],conv_info['W'],conv_info['F'],conv_info['K'],conv_info['S'],conv_info['D'],conv_info['P'], conv_info['HO'], conv_info['WO'])
    f16_log_path = os.path.join(log_dir, f16_log_name)
    # print(f16_log_path)
    _algo, _min_runtime = parse_algo_and_runtime(f16_log_path)
    print(_algo)

