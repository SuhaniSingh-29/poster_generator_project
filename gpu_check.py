import torch
print("CUDA available:", torch.cuda.is_available())
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
