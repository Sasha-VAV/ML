from re import I

import torch
import softmax

input = torch.tensor([2.1, 2.0, 2.0, 8.8], requires_grad=True)
sm = torch.nn.Softmax(dim=0)
output = sm(input)
print(f"PyTorch forward {output}")
#output.sum().backward()
#print(input)
input = torch.tensor([2.1, 2.0, 2.0, 8.8], requires_grad=True)
sm = softmax.Softmax.apply
output = sm(input)
ground_true = torch.tensor([0., 0., 0., 1.])
print(f"My softmax forward res: {output}")


loss = (output - ground_true) ** 2
print(f"My softmax forward loss: {loss}")
ground_true = torch.tensor([1., 1., 0., 1.])
loss = (output - ground_true) ** 2
loss = loss.sum().backward()
print(f"My softmax grad: {input.grad}")


input = torch.tensor([2.1, 2.0, 2.0, 8.8], requires_grad=True)
sm = torch.nn.Softmax(dim=0)
output = sm(input)
ground_true = torch.tensor([1., 1., 0., 1.])
#print(input.grad)

loss = (output - ground_true) ** 2
loss = loss.sum().backward()
print(f"Torch softmax grad: {input.grad}")


input = torch.tensor([4.1, 6.0, 2.0, 1.8], requires_grad=True)
sm = softmax.Softmax.apply
output = sm(input)

ground_true = torch.tensor([1., 1., 0., 1.])
loss = (output - ground_true) ** 2
loss = loss.sum().backward()
print(f"My softmax grad: {input.grad}")


input = torch.tensor([4.1, 6.0, 2.0, 1.8], requires_grad=True)
sm = torch.nn.Softmax(dim=0)
output = sm(input)
ground_true = torch.tensor([1., 1., 0., 1.])
#print(input.grad)

loss = (output - ground_true) ** 2
loss = loss.sum().backward()
print(f"Torch softmax grad: {input.grad}")