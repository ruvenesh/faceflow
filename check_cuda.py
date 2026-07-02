import torch
print('Version:', torch.__version__)
print('CUDA toolkit:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('Arch list:', torch.cuda.get_arch_list())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('Capability:', torch.cuda.get_device_capability(0))
    t = torch.randn(3, 3).cuda()
    print('Test tensor OK:', t.shape)
