import torch

weights_f32 = torch.rand((10, 3), dtype=torch.float32)
input_f32 = torch.rand(10, dtype=torch.float32)
w_min = weights_f32.min()
w_max = weights_f32.max()
scale = (w_max - w_min) / 255
z = torch.tensor(0)  # torch.round(-w_min / scale)
weights_int8 = (torch.round(weights_f32 / scale) + z).clamp(0,255).to(dtype=torch.uint8)
input_int8 = (input_f32 * 255).round().to(dtype=torch.uint8)
x_f32 = input_f32 @ weights_f32
x_int32 = input_int8.float() @ weights_int8.float()
x_f32_q = (x_int32 * scale / 255)
print(x_f32, x_f32_q)
