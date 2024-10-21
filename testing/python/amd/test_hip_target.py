import tvm

# 定义目标后端名称
target_backend = "hip"

# 尝试创建目标对象
try:
    target = tvm.target.Target(target_backend)
    print(f"Success: '{target_backend}' backend is available in TVM.")
except ValueError as e:
    print(f"Error: '{target_backend}' backend is not available in TVM. Details: {e}")