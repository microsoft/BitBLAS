import re
import os

log_list_e5m2 = [
    # 'logs/resnet-18_fp8_e5m2.log',
    "logs/resnet-50_fp8_e5m2.log",
    # 'logs/mobilenet-v2_fp8_e5m2.log',
    # 'logs/VGG-16_fp8_e5m2.log',
    # 'logs/NAFNet_fp8_e5m2.log',
    "logs/shufflenet_fp8_e5m2.log",
    # 'logs/squeezenet_fp8_e5m2.log',
    "logs/Conformer_fp8_e5m2.log",
    "logs/vit_fp8_e5m2.log",
    # 'logs/mobilevit_fp8_e5m2.log',
]


log_list_int4xint4 = [
    # 'logs/resnet-18_fp8_int4b.log',
    "logs/resnet-50_fp8_int4b.log",
    # 'logs/mobilenet-v2_fp8_int4b.log',
    # 'logs/VGG-16_fp8_int4b.log',
    # 'logs/NAFNet_fp8_int4b.log',
    "logs/shufflenet_fp8_int4b.log",
    # 'logs/squeezenet_fp8_int4b.log',
    "logs/Conformer_fp8_int4b.log",
    "logs/vit_fp8_int4b.log",
    # 'logs/mobilevit_fp8_int4b.log',
]

log_list_int4xint1 = [
    # 'logs/resnet-18_fp8_int4bxint1.log',
    "logs/resnet-50_fp8_int4bxint1.log",
    # 'logs/mobilenet-v2_fp8_int4bxint1.log',
    # 'logs/VGG-16_fp8_int4bxint1.log',
    # 'logs/NAFNet_fp8_int4bxint1.log',
    "logs/shufflenet_fp8_int4bxint1.log",
    # 'logs/squeezenet_fp8_int4bxint1.log',
    "logs/Conformer_fp8_int4bxint1.log",
    "logs/vit_fp8_int4bxint1.log",
    # 'logs/mobilevit_fp8_int4bxint1.log',
]

log_list_mxfp8 = [
    # 'logs/resnet-18_fp8_mxfp8_e5m2.log',
    "logs/resnet-50_fp8_mxfp8_e5m2.log",
    # 'logs/mobilenet-v2_fp8_mxfp8_e5m2.log',
    # 'logs/VGG-16_fp8_mxfp8_e5m2.log',
    # 'logs/NAFNet_fp8_mxfp8_e5m2.log',
    "logs/shufflenet_fp8_mxfp8_e5m2.log",
    # 'logs/squeezenet_fp8_mxfp8_e5m2.log',
    "logs/Conformer_fp8_mxfp8_e5m2.log",
    "logs/vit_fp8_mxfp8_e5m2.log",
    # 'logs/mobilevit_fp8_mxfp8_e5m2.log',
]


def extract_mean_value(text):
    pattern = r"[\d]+\.[\d]+"

    matches = re.findall(pattern, text)

    if matches and len(matches) >= 4:
        return float(matches[-4])
    else:
        return None


mean_times = []
logs = log_list_e5m2
logs = log_list_int4xint4
logs = log_list_int4xint1
logs = log_list_mxfp8
for log in logs:
    print(log)
    # check if file exists
    if not os.path.exists(log):
        mean_times.append(1000.0)
        continue

    with open(log, "r") as f:
        lines = f.read()
        mean_times.append(extract_mean_value(lines))

print(" ".join([str(t) for t in mean_times]))
