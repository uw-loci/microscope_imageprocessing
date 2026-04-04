import cupy as cp

cp.show_config()  # should list CUDA/ROCm and your device
print("GPU count:", cp.cuda.runtime.getDeviceCount())
props = cp.cuda.runtime.getDeviceProperties(0)
print("Using:", props["name"].decode())
# quick test
x = cp.arange(10**6, dtype=cp.float32)
print((x * x).sum())  # runs on GPU if the install is correct
