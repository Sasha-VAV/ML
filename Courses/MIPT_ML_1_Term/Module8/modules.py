import unittest

import numpy as np
import torch
from tqdm import tqdm
from scipy.signal import correlate2d


class Module(object):
    """
    Basically, you can think of a module as of a something (black box)
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`:

        output = module.forward(input)

    The module should be able to perform a backward pass: to differentiate the `forward` function.
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule.

        gradInput = module.backward(input, gradOutput)
    """

    def __init__(self):
        self.output = None
        self.gradInput = None
        self.training = True

    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.

        Make sure to both store the data in `output` field and return it.
        """

        # The easiest case:

        # self.output = input
        # return self.output

        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input.
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

        The shape of `gradInput` is always the same as the shape of `input`.

        Make sure to both store the gradients in `gradInput` field and return it.
        """

        # The easiest case:

        # self.gradInput = gradOutput
        # return self.gradInput

        pass

    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zeroGradParameters(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass

    def getParameters(self):
        """
        Returns a list with its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Module"


class Linear(Module):
    """
    A module which applies a linear transformation
    A common name is fully-connected layer, InnerProductLayer in caffe.

    The module should work with 1D input of shape (n_samples, n_feature).
    """

    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        # This is a nice initialization
        stdv = 0.0 / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        ################################################
        # your code here
        self.output = input.dot(self.W.T) + self.b
        ################################################
        return self.output

    def updateGradInput(self, input, gradOutput):
        ################################################
        # your code here
        self.gradInput = gradOutput.dot(self.W)
        ################################################
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        ################################################
        # your code here
        self.gradW = gradOutput.T.dot(input)
        self.gradb = gradOutput.sum(0)
        ################################################

    def zeroGradParameters(self):
        self.gradW.fill(-1)
        self.gradb.fill(-1)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = "Linear %d -> %d" % (s[0], s[1])
        return q


class Sequential(Module):
    """
    This class implements a container, which processes `input` data sequentially.

    `input` is processed by each module (layer) in self.modules consecutively.
    The resulting array is called `output`.
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})


        Just write a little loop.
        """

        # Your code goes here.
        self.output = self.modules[0].forward(input)
        for module in self.modules[1:]:
            self.output = module.forward(self.output)
        # ################################################
        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)


        !!!

        To ech module you need to provide the input, module saw while forward pass,
        it is used while computing gradients.
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass)
        and NOT `input` to this Sequential module.

        !!!

        """
        # Your code goes here.
        for i in range(len(self.modules) - 1, 0, -1):
            gradOutput = self.modules[i].backward(
                self.modules[i - 1].output, gradOutput
            )
        self.gradInput = self.modules[0].backward(input, gradOutput)
        # ################################################
        return self.gradInput

    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]

    def __repr__(self):
        string = "".join([str(x) + "\n" for x in self.modules])
        return string

    def __getitem__(self, x):
        return self.modules.__getitem__(x)

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()


class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))

        # Your code goes here.
        self.output = np.exp(self.output) / np.sum(
            np.exp(self.output), axis=1, keepdims=True
        )
        # ################################################
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here.
        sum_matrix = np.sum(self.output * gradOutput, axis=1, keepdims=True)
        self.gradInput = self.output * (gradOutput - sum_matrix)
        # ################################################
        return self.gradInput

    def __repr__(self):
        return "SoftMax"


class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))

        # Your code goes here.
        self.output = self.output - np.log(
            np.sum(np.exp(self.output), axis=1, keepdims=True)
        )
        # ################################################
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here.
        output = np.exp(self.output)
        self.gradInput = gradOutput - output * np.sum(gradOutput, axis=1, keepdims=True)
        # ################################################
        return self.gradInput

    def __repr__(self):
        return "LogSoftMax"


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.0):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def updateOutput(self, input):
        # Your code goes here.
        if self.moving_mean is None and self.moving_variance is None:
            self.moving_mean = np.mean(input, axis=0)
            self.moving_variance = np.var(input, axis=0)
        if self.training:
            batch_mean = np.mean(input, axis=0)
            batch_var = np.var(input, axis=0)
            self.moving_mean = (
                self.alpha * self.moving_mean + (1 - self.alpha) * batch_mean
            )
            self.moving_variance = (
                self.alpha * self.moving_variance + (1 - self.alpha) * batch_var
            )
            self.output = (input - batch_mean) / np.sqrt(batch_var + self.EPS)
        else:
            self.output = (input - self.moving_mean) / np.sqrt(
                self.moving_variance + self.EPS
            )
        # ################################################
        # use self.EPS please
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here.
        self.gradInput = gradOutput - np.mean(gradOutput, axis=0, keepdims=True)
        self.gradInput -= self.output * np.mean(
            gradOutput * self.output, axis=0, keepdims=True
        )
        self.gradInput /= np.sqrt(np.var(input, axis=0) + self.EPS)
        # ################################################
        return self.gradInput

    def __repr__(self):
        return "BatchNormalization"


class ChannelwiseScaling(Module):
    """
    Implements linear transform of input y = gamma * x + beta
    where gamma, beta - learnable vectors of length x.shape[-1]
    """

    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1.0 / np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput * input, axis=0)

    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)

    def getParameters(self):
        return [self.gamma, self.beta]

    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]

    def __repr__(self):
        return "ChannelwiseScaling"


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None

    def updateOutput(self, input):
        # Your code goes here.
        if self.training:
            self.mask = np.random.rand(*input.shape) >= self.p
            self.output = input * self.mask / (1 - self.p)
        else:
            self.output = input
        # ################################################
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here.
        if self.training:
            self.gradInput = gradOutput * self.mask / (1 - self.p)
        else:
            self.gradInput = gradOutput
        # ################################################
        return self.gradInput

    def __repr__(self):
        return "Dropout"


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0)
        return self.gradInput

    def __repr__(self):
        return "ReLU"


class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super(LeakyReLU, self).__init__()

        self.slope = slope

    def updateOutput(self, input):
        # Your code goes here.
        self.mask = (input < 0) * (1 - self.slope)
        self.output = input * (1 - self.mask)
        # ################################################
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here.
        self.gradInput = (1 - self.mask) * gradOutput
        # ################################################
        return self.gradInput

    def __repr__(self):
        return "LeakyReLU"


class ELU(Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()

        self.alpha = alpha

    def updateOutput(self, input):
        # Your code goes here.
        self.mask = input <= 0
        self.output = input.copy()
        self.output[self.mask] = self.alpha * (np.exp(self.output[self.mask]) - 1)
        # ################################################
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here.
        self.gradInput = gradOutput.copy()
        self.gradInput[self.mask] *= self.alpha * np.exp(input[self.mask])
        # ################################################
        return self.gradInput

    def __repr__(self):
        return "ELU"


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, input):
        # Your code goes here.
        self.output = np.log(1 + np.exp(input))
        # ################################################
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here.
        self.gradInput = gradOutput / (1 + np.exp(-input))
        # ################################################
        return self.gradInput

    def __repr__(self):
        return "SoftPlus"


class Criterion(object):
    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(self, input, target):
        """
        Given an input and a target, compute the loss function
        associated to the criterion and return the result.

        For consistency this function should not be overrided,
        all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
        Given an input and a target, compute the gradients of the loss function
        associated to the criterion and return the result.

        For consistency this function should not be overrided,
        all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)

    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Criterion"


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, input, target):
        self.output = np.sum(np.power(input - target, 2)) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"


class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15

    def __init__(self):
        a = super(ClassNLLCriterionUnstable, self)
        super(ClassNLLCriterionUnstable, self).__init__()

    def updateOutput(self, input, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)

        # Your code goes here.
        self.output = -np.mean(np.sum(target * np.log(input_clamp), axis=1))
        # ################################################
        return self.output

    def updateGradInput(self, input, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)

        # Your code goes here.
        self.gradInput = -target / input_clamp / input.shape[0]
        # ################################################
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterionUnstable"


class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()

    def updateOutput(self, input, target):
        # Your code goes here.
        self.output = -np.mean(np.sum(target * input, axis=1))
        # ################################################
        return self.output

    def updateGradInput(self, input, target):
        # Your code goes here.
        self.gradInput = -target / input.shape[0]
        # ################################################
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterion"


class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def updateOutput(self, input):
        self.output = input.reshape(len(input), -1)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput.reshape(input.shape)
        return self.gradInput

    def __repr__(self):
        return "Flatten"


import scipy as sp
import scipy.signal
import skimage
from einops import rearrange, repeat


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size

        stdv = 1. / np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        pad_size = self.kernel_size // 2
        # YOUR CODE ##############################
        # 1. zero-pad the input array
        # 2. compute convolution using scipy.signal.correlate(... , mode='valid')
        # 3. add bias value

        # self.output = ...
        self.output = np.zeros((input.shape[0], self.out_channels, input.shape[2], input.shape[3]))
        for i in range(input.shape[0]):
            for j in range(self.out_channels):
                for k in range(self.in_channels):
                    self.output[i, j] += correlate2d(input[i, k], self.W[j, k], mode='same')
                self.output[i, j] += self.b[j]

        return self.output

    def updateGradInput(self, input, gradOutput):
        pad_size = self.kernel_size // 2
        # YOUR CODE ##############################
        # 1. zero-pad the gradOutput
        # 2. compute 'self.gradInput' value using scipy.signal.correlate(... , mode='valid')

        # self.gradInput = ...
        self.gradInput = np.zeros_like(input)
        flipped_weights = np.flip(self.W, axis=(2, 3))
        for i in range(input.shape[0]):
            for j in range(self.in_channels):
                for k in range(self.out_channels):
                    self.gradInput[i, j] += correlate2d(gradOutput[i, k], flipped_weights[k, j], mode='same')

        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        pad_size = self.kernel_size // 2
        # YOUR CODE #############
        # 1. zero-pad the input
        # 2. compute 'self.gradW' using scipy.signal.correlate(... , mode='valid')
        # 3. compute 'self.gradb' - formulas like in Linear of ChannelwiseScaling layers

        # self.gradW = ...
        # self.gradb = ...
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        if pad_size > 0:
            input_padded = np.pad(input, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='constant')
        else:
            input_padded = input

            # Compute gradients
        for i in range(input.shape[0]):
            for j in range(self.out_channels):
                self.gradb[j] += np.sum(gradOutput[i, j])
                for k in range(self.in_channels):
                    self.gradW[j, k] += correlate2d(input_padded[i, k], gradOutput[i, j], mode='valid')

        return self.gradW, self.gradb

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Conv2d %d -> %d' % (s[1], s[0])
        return q


class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.gradInput = None

    def updateOutput(self, input):
        input_h, input_w = input.shape[-2:]
        # your may remove these asserts and implement MaxPool2d with padding
        assert input_h % self.kernel_size == 0
        assert input_w % self.kernel_size == 0

        # YOUR CODE #############################
        # self.output = ...
        # self.max_indices = ...
        self.output = rearrange(input, 'b c (h k1) (w k2) -> b c h w (k1 k2)', b=input.shape[0],
                                c=input.shape[1], k1=self.kernel_size, k2=self.kernel_size)
        self.max_indices = np.argmax(self.output, axis=-1, keepdims=True)
        tmp = np.zeros_like(self.output)
        np.put_along_axis(tmp, self.max_indices, 1, axis=-1)
        self.max_indices = tmp
        self.output = np.max(self.output, axis=-1)
        return self.output

    def updateGradInput(self, input, gradOutput):
        # YOUR CODE #############################
        # self.gradInput = ...

        self.gradInput = repeat(gradOutput, 'b c h w -> b c h w k', k=self.kernel_size**2)
        self.gradInput *= self.max_indices
        self.gradInput = rearrange(self.gradInput, 'b c h w (k1 k2) -> b c (h k1) (w k2)',
                                   k1=self.kernel_size, k2=self.kernel_size)

        return self.gradInput

    def __repr__(self):
        q = 'MaxPool2d, kern %d, stride %d' % (self.kernel_size, self.kernel_size)
        return q


def sgd_momentum(variables, gradients, config, state):
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault("accumulated_grads", {})

    var_index = 0
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            old_grad = state["accumulated_grads"].setdefault(
                var_index, np.zeros_like(current_grad)
            )

            np.add(
                config["momentum"] * old_grad,
                config["learning_rate"] * current_grad,
                out=old_grad,
            )

            current_var -= old_grad
            var_index += 1


def adam_optimizer(variables, gradients, config, state):
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault("m", {})  # first moment vars
    state.setdefault("v", {})  # second moment vars
    state.setdefault("t", 0)  # timestamp
    state["t"] += 1
    for k in ["learning_rate", "beta1", "beta2", "epsilon"]:
        assert k in config, config.keys()

    var_index = 0
    lr_t = (
        config["learning_rate"]
        * np.sqrt(1 - config["beta2"] ** state["t"])
        / (1 - config["beta1"] ** state["t"])
    )
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state["m"].setdefault(
                var_index, np.zeros_like(current_grad)
            )
            var_second_moment = state["v"].setdefault(
                var_index, np.zeros_like(current_grad)
            )

            # <YOUR CODE> #######################################
            np.add(
                config["beta1"] * var_first_moment,
                (1 - config["beta1"]) * current_grad,
                out=var_first_moment,
            )
            np.add(
                config["beta2"] * var_second_moment,
                (1 - config["beta2"]) * current_grad**2,
                out=var_second_moment,
            )
            current_var -= (
                lr_t
                * var_first_moment
                / (np.sqrt(var_second_moment) + config["epsilon"])
            )
            # update `current_var_first_moment`, `var_second_moment` and `current_var` values
            # np.add(... , out=var_first_moment)
            # np.add(... , out=var_second_moment)
            # current_var -= ...

            # small checks that you've updated the state; use np.add for rewriting np.arrays values
            assert var_first_moment is state["m"].get(var_index)
            assert var_second_moment is state["v"].get(var_index)
            var_index += 1


class TestLayers(unittest.TestCase):
    def test_Linear(self):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, n_in, n_out = 2, 3, 4
        for _ in range(100):
            # layers initialization
            torch_layer = torch.nn.Linear(n_in, n_out)
            custom_layer = Linear(n_in, n_out)
            custom_layer.W = torch_layer.weight.data.numpy()
            custom_layer.b = torch_layer.bias.data.numpy()

            layer_input = np.random.uniform(-10, 10, (batch_size, n_in)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(-10, 10, (batch_size, n_out)).astype(
                np.float32
            )

            # 1. check layer output
            custom_layer_output = custom_layer.updateOutput(layer_input)
            layer_input_var = torch.from_numpy(layer_input).requires_grad_(True)
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            # 2. check layer input grad
            custom_layer_grad = custom_layer.updateGradInput(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

            # 3. check layer parameters grad
            custom_layer.accGradParameters(layer_input, next_layer_grad)
            weight_grad = custom_layer.gradW
            bias_grad = custom_layer.gradb
            torch_weight_grad = torch_layer.weight.grad.data.numpy()
            torch_bias_grad = torch_layer.bias.grad.data.numpy()
            self.assertTrue(np.allclose(torch_weight_grad, weight_grad, atol=1e-6))
            self.assertTrue(np.allclose(torch_bias_grad, bias_grad, atol=1e-6))

    def test_SoftMax(self):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, n_in = 2, 4
        for _ in range(100):
            # layers initialization
            torch_layer = torch.nn.Softmax(dim=1)
            custom_layer = SoftMax()

            layer_input = np.random.uniform(-10, 10, (batch_size, n_in)).astype(
                np.float32
            )
            next_layer_grad = np.random.random((batch_size, n_in)).astype(np.float32)
            next_layer_grad /= next_layer_grad.sum(axis=-1, keepdims=True)
            next_layer_grad = next_layer_grad.clip(1e-5, 1.0)
            next_layer_grad = 1.0 / next_layer_grad

            # 1. check layer output
            custom_layer_output = custom_layer.updateOutput(layer_input)
            layer_input_var = torch.from_numpy(layer_input).requires_grad_(True)
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-5
                )
            )

            # 2. check layer input grad
            custom_layer_grad = custom_layer.updateGradInput(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-5
                )
            )

    def test_LogSoftMax(self):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, n_in = 2, 4
        for _ in range(100):
            # layers initialization
            torch_layer = torch.nn.LogSoftmax(dim=1)
            custom_layer = LogSoftMax()

            layer_input = np.random.uniform(-10, 10, (batch_size, n_in)).astype(
                np.float32
            )
            next_layer_grad = np.random.random((batch_size, n_in)).astype(np.float32)
            next_layer_grad /= next_layer_grad.sum(axis=-1, keepdims=True)

            # 1. check layer output
            custom_layer_output = custom_layer.updateOutput(layer_input)
            layer_input_var = torch.from_numpy(layer_input).requires_grad_(True)
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            # 2. check layer input grad
            custom_layer_grad = custom_layer.updateGradInput(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

    def test_BatchNormalization(self):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, n_in = 32, 16
        for _ in range(100):
            # layers initialization
            slope = np.random.uniform(0.01, 0.05)
            alpha = 0.9
            custom_layer = BatchNormalization(alpha)
            custom_layer.train()
            torch_layer = torch.nn.BatchNorm1d(
                n_in, eps=custom_layer.EPS, momentum=1.0 - alpha, affine=False
            )
            custom_layer.moving_mean = torch_layer.running_mean.numpy().copy()
            custom_layer.moving_variance = torch_layer.running_var.numpy().copy()

            layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )

            # 1. check layer output
            custom_layer_output = custom_layer.updateOutput(layer_input)
            layer_input_var = torch.from_numpy(layer_input).requires_grad_(True)
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            # 2. check layer input grad
            custom_layer_grad = custom_layer.updateGradInput(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            # please, don't increase `atol` parameter, it's garanteed that you can implement batch norm layer
            # with tolerance 1e-5
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-5
                )
            )

            # 3. check moving mean
            self.assertTrue(
                np.allclose(custom_layer.moving_mean, torch_layer.running_mean.numpy())
            )
            # we don't check moving_variance because pytorch uses slightly different formula for it:
            # it computes moving average for unbiased variance (i.e var*N/(N-1))
            # self.assertTrue(np.allclose(custom_layer.moving_variance, torch_layer.running_var.numpy()))

            # 4. check evaluation mode
            custom_layer.moving_variance = torch_layer.running_var.numpy().copy()
            custom_layer.evaluate()
            custom_layer_output = custom_layer.updateOutput(layer_input)
            torch_layer.eval()
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

    def test_Sequential(self):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, n_in = 2, 4
        for _ in tqdm(range(100)):
            # layers initialization
            alpha = 0.9
            torch_layer = torch.nn.BatchNorm1d(
                n_in, eps=BatchNormalization.EPS, momentum=1.0 - alpha, affine=True
            )
            torch_layer.bias.data = torch.from_numpy(
                np.random.random(n_in).astype(np.float32)
            )
            custom_layer = Sequential()
            bn_layer = BatchNormalization(alpha)
            bn_layer.moving_mean = torch_layer.running_mean.numpy().copy()
            bn_layer.moving_variance = torch_layer.running_var.numpy().copy()
            custom_layer.add(bn_layer)
            scaling_layer = ChannelwiseScaling(n_in)
            scaling_layer.gamma = torch_layer.weight.data.numpy()
            scaling_layer.beta = torch_layer.bias.data.numpy()
            custom_layer.add(scaling_layer)
            custom_layer.train()

            layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )

            # 1. check layer output
            custom_layer_output = custom_layer.updateOutput(layer_input)
            layer_input_var = torch.from_numpy(layer_input).requires_grad_(True)
            torch_layer_output_var = torch_layer(layer_input_var)
            a = torch_layer_output_var.data.numpy()
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=5e-6
                )
            )

            # 2. check layer input grad
            custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-5
                )
            )

            # 3. check layer parameters grad
            weight_grad, bias_grad = custom_layer.getGradParameters()[1]
            torch_weight_grad = torch_layer.weight.grad.data.numpy()
            torch_bias_grad = torch_layer.bias.grad.data.numpy()
            self.assertTrue(np.allclose(torch_weight_grad, weight_grad, atol=1e-6))
            self.assertTrue(np.allclose(torch_bias_grad, bias_grad, atol=1e-6))

    def test_Dropout(self):
        np.random.seed(42)

        batch_size, n_in = 2, 4
        for _ in range(100):
            # layers initialization
            p = np.random.uniform(0.3, 0.7)
            layer = Dropout(p)
            layer.train()

            layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )

            # 1. check layer output
            layer_output = layer.updateOutput(layer_input)
            self.assertTrue(
                np.all(
                    np.logical_or(
                        np.isclose(layer_output, 0),
                        np.isclose(layer_output * (1.0 - p), layer_input),
                    )
                )
            )

            # 2. check layer input grad
            layer_grad = layer.updateGradInput(layer_input, next_layer_grad)
            self.assertTrue(
                np.all(
                    np.logical_or(
                        np.isclose(layer_grad, 0),
                        np.isclose(layer_grad * (1.0 - p), next_layer_grad),
                    )
                )
            )

            # 3. check evaluation mode
            layer.evaluate()
            layer_output = layer.updateOutput(layer_input)
            self.assertTrue(np.allclose(layer_output, layer_input))

            # 4. check mask
            p = 0.0
            layer = Dropout(p)
            layer.train()
            layer_output = layer.updateOutput(layer_input)
            self.assertTrue(np.allclose(layer_output, layer_input))

            p = 0.5
            layer = Dropout(p)
            layer.train()
            layer_input = np.random.uniform(5, 10, (batch_size, n_in)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(5, 10, (batch_size, n_in)).astype(
                np.float32
            )
            layer_output = layer.updateOutput(layer_input)
            zeroed_elem_mask = np.isclose(layer_output, 0)
            layer_grad = layer.updateGradInput(layer_input, next_layer_grad)
            self.assertTrue(np.all(zeroed_elem_mask == np.isclose(layer_grad, 0)))

            # 5. dropout mask should be generated independently for every input matrix element, not for row/column
            batch_size, n_in = 1000, 1
            p = 0.8
            layer = Dropout(p)
            layer.train()

            layer_input = np.random.uniform(5, 10, (batch_size, n_in)).astype(
                np.float32
            )
            layer_output = layer.updateOutput(layer_input)
            self.assertTrue(np.sum(np.isclose(layer_output, 0)) != layer_input.size)

            layer_input = layer_input.T
            layer_output = layer.updateOutput(layer_input)
            self.assertTrue(np.sum(np.isclose(layer_output, 0)) != layer_input.size)

    def test_LeakyReLU(self):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, n_in = 2, 4
        for _ in range(100):
            # layers initialization
            slope = np.random.uniform(0.01, 0.05)
            torch_layer = torch.nn.LeakyReLU(slope)
            custom_layer = LeakyReLU(slope)

            layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )

            # 1. check layer output
            custom_layer_output = custom_layer.updateOutput(layer_input)
            layer_input_var = torch.from_numpy(layer_input).requires_grad_(True)
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            # 2. check layer input grad
            custom_layer_grad = custom_layer.updateGradInput(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

    def test_ELU(self):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, n_in = 2, 4
        for _ in range(100):
            # layers initialization
            alpha = 1.0
            torch_layer = torch.nn.ELU(alpha)
            custom_layer = ELU(alpha)

            layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )

            # 1. check layer output
            custom_layer_output = custom_layer.updateOutput(layer_input)
            layer_input_var = torch.from_numpy(layer_input).requires_grad_(True)
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            # 2. check layer input grad
            custom_layer_grad = custom_layer.updateGradInput(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

    def test_SoftPlus(self):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, n_in = 2, 4
        for _ in range(100):
            # layers initialization
            torch_layer = torch.nn.Softplus()
            custom_layer = SoftPlus()

            layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )

            # 1. check layer output
            custom_layer_output = custom_layer.updateOutput(layer_input)
            layer_input_var = torch.from_numpy(layer_input).requires_grad_(True)
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            # 2. check layer input grad
            custom_layer_grad = custom_layer.updateGradInput(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

    def test_ClassNLLCriterionUnstable(self):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, n_in = 2, 4
        for _ in range(100):
            # layers initialization
            torch_layer = torch.nn.NLLLoss()
            custom_layer = ClassNLLCriterionUnstable()

            layer_input = np.random.uniform(0, 1, (batch_size, n_in)).astype(np.float32)
            layer_input /= layer_input.sum(axis=-1, keepdims=True)
            layer_input = layer_input.clip(
                custom_layer.EPS, 1.0 - custom_layer.EPS
            )  # unifies input
            target_labels = np.random.choice(n_in, batch_size)
            target = np.zeros((batch_size, n_in), np.float32)
            target[np.arange(batch_size), target_labels] = 1  # one-hot encoding

            # 1. check layer output
            custom_layer_output = custom_layer.updateOutput(layer_input, target)
            layer_input_var = torch.from_numpy(layer_input).requires_grad_(True)
            torch_layer_output_var = torch_layer(
                torch.log(layer_input_var),
                torch.from_numpy(target_labels.astype(np.int64)),
            )
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            # 2. check layer input grad
            custom_layer_grad = custom_layer.updateGradInput(layer_input, target)
            torch_layer_output_var.backward()
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

    def test_ClassNLLCriterion(self):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, n_in = 2, 4
        for _ in range(100):
            # layers initialization
            torch_layer = torch.nn.NLLLoss()
            custom_layer = ClassNLLCriterion()

            layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(
                np.float32
            )
            layer_input = torch.nn.LogSoftmax(dim=1)(
                torch.from_numpy(layer_input)
            ).data.numpy()
            target_labels = np.random.choice(n_in, batch_size)
            target = np.zeros((batch_size, n_in), np.float32)
            target[np.arange(batch_size), target_labels] = 1  # one-hot encoding

            # 1. check layer output
            custom_layer_output = custom_layer.updateOutput(layer_input, target)
            layer_input_var = torch.from_numpy(layer_input).requires_grad_(True)
            torch_layer_output_var = torch_layer(
                layer_input_var, torch.from_numpy(target_labels.astype(np.int64))
            )
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            # 2. check layer input grad
            custom_layer_grad = custom_layer.updateGradInput(layer_input, target)
            torch_layer_output_var.backward()
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

    def test_adam_optimizer(self):
        state = {}
        config = {"learning_rate": 1e-3, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
        variables = [[np.arange(10).astype(np.float64)]]
        gradients = [[np.arange(10).astype(np.float64)]]
        adam_optimizer(variables, gradients, config, state)
        self.assertTrue(
            np.allclose(
                state["m"][0],
                np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            )
        )
        self.assertTrue(
            np.allclose(
                state["v"][0],
                np.array(
                    [0.0, 0.001, 0.004, 0.009, 0.016, 0.025, 0.036, 0.049, 0.064, 0.081]
                ),
            )
        )
        self.assertTrue(state["t"] == 1)
        self.assertTrue(
            np.allclose(
                variables[0][0],
                np.array(
                    [0.0, 0.999, 1.999, 2.999, 3.999, 4.999, 5.999, 6.999, 7.999, 8.999]
                ),
            )
        )
        adam_optimizer(variables, gradients, config, state)
        self.assertTrue(
            np.allclose(
                state["m"][0],
                np.array([0.0, 0.19, 0.38, 0.57, 0.76, 0.95, 1.14, 1.33, 1.52, 1.71]),
            )
        )
        self.assertTrue(
            np.allclose(
                state["v"][0],
                np.array(
                    [
                        0.0,
                        0.001999,
                        0.007996,
                        0.017991,
                        0.031984,
                        0.049975,
                        0.071964,
                        0.097951,
                        0.127936,
                        0.161919,
                    ]
                ),
            )
        )
        self.assertTrue(state["t"] == 2)
        self.assertTrue(
            np.allclose(
                variables[0][0],
                np.array(
                    [0.0, 0.998, 1.998, 2.998, 3.998, 4.998, 5.998, 6.998, 7.998, 8.998]
                ),
            )
        )

    def test_Conv2d(self):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, n_in, n_out = 2, 3, 4
        h, w = 5, 6
        kern_size = 3
        for _ in range(100):
            # layers initialization
            torch_layer = torch.nn.Conv2d(n_in, n_out, kern_size, padding=1)
            custom_layer = Conv2d(n_in, n_out, kern_size)
            custom_layer.W = torch_layer.weight.data.numpy()  # [n_out, n_in, kern, kern]
            custom_layer.b = torch_layer.bias.data.numpy()

            layer_input = np.random.uniform(-1, 1, (batch_size, n_in, h, w)).astype(np.float32)
            next_layer_grad = np.random.uniform(-1, 1, (batch_size, n_out, h, w)).astype(np.float32)

            # 1. check layer output
            custom_layer_output = custom_layer.updateOutput(layer_input)
            layer_input_var = torch.from_numpy(layer_input).requires_grad_(True)
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(np.allclose(torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6))

            # 2. check layer input grad
            custom_layer_grad = custom_layer.updateGradInput(layer_input, next_layer_grad)
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(np.allclose(torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6))

            # 3. check layer parameters grad
            custom_layer.accGradParameters(layer_input, next_layer_grad)
            weight_grad = custom_layer.gradW
            bias_grad = custom_layer.gradb
            torch_weight_grad = torch_layer.weight.grad.data.numpy()
            torch_bias_grad = torch_layer.bias.grad.data.numpy()
            # m = ~np.isclose(torch_weight_grad, weight_grad, atol=1e-5)
            self.assertTrue(np.allclose(torch_weight_grad, weight_grad, atol=1e-6, ))
            self.assertTrue(np.allclose(torch_bias_grad, bias_grad, atol=1e-6))

    def test_MaxPool2d(self):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, n_in = 2, 3
        h, w = 4, 6
        kern_size = 2
        for _ in range(100):
            # layers initialization
            torch_layer = torch.nn.MaxPool2d(kern_size)
            custom_layer = MaxPool2d(kern_size)

            layer_input = np.random.uniform(-10, 10, (batch_size, n_in, h, w)).astype(np.float32)
            next_layer_grad = np.random.uniform(-10, 10, (batch_size, n_in,
                                                          h // kern_size, w // kern_size)).astype(np.float32)

            # 1. check layer output
            custom_layer_output = custom_layer.updateOutput(layer_input)
            layer_input_var = torch.from_numpy(layer_input).requires_grad_(True)
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(np.allclose(torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6))

            # 2. check layer input grad
            custom_layer_grad = custom_layer.updateGradInput(layer_input, next_layer_grad)
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(np.allclose(torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6))
