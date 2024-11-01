import torch
import math
import numpy as np


class Softmax(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        sum_j_k = sum(math.exp(input[j]) for j in range(len(input)))
        probability = torch.tensor(list(math.exp(input[i]) / sum_j_k for i in range(len(input))))
        return probability

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        sum_j_k = sum(math.exp(input[j]) for j in range(len(input)))
        probability = np.array([math.exp(input[i]) / sum_j_k for i in range(len(input))])
        delta = 1e-3
        new_input = tuple(input[i] + delta for i in range(len(input)))
        new_sum_j_k = sum(math.exp(new_input[i]) for i in range(len(input)))
        new_probability = np.array([math.exp(new_input[i]) / new_sum_j_k for i in range(len(input))])
        res = torch.tensor(list((new_probability[i] - probability[i]) / delta for i in range(len(input))))
        return res

