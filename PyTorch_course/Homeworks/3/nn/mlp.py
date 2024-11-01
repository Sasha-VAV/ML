import random
from autograd import Value
import numpy as np


class Module():
    def zero_grad(self):
        self.grad = 0

    def parameters(self):
        return self.data


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = np.array([Value() for _ in range(nin)])
        self.b = Value()
        self.nonlin = nonlin

    def __call__(self, x):
        act = self.b
        for i in range(len(self.w)):
            act += self.w[i] * x[i]
        self.data = act
        if self.nonlin:
            act.data = max(act.data, 0)
            return act
        return act

    def parameters(self):
        return tuple(self.w[i] for i in range(len(self.w))) + (self.b,)

    def zero_grad(self):
        for w in self.w:
            w.grad = 0
        self.b.grad = 0

    def backward(self):
        self.data.backward()

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        vals = list(Neuron(nin, kwargs['nonlin' if kwargs.__contains__('nonlin') else None]) for _ in range(nout))
        self.neurons = np.array(vals, dtype=object)

    def __call__(self, x):
        out = list(self.neurons[i](x) for i in range(len(self.neurons)))
        return out[0] if len(out) == 1 else out

    def zero_grad(self):
        for x in self.neurons:
            x.zero_grad()

    def parameters(self):
        return tuple(param for i in range(len(self.neurons)) for param in self.neurons[i].parameters())

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, nouts):
        sz = nouts.copy()
        sz.insert(0, nin)

        self.layers = [Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self):
        for neuron in self.layers[-1].neurons:
            neuron.backward()

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def parameters(self):
        return tuple(param for i in range(len(self.layers)) for param in self.layers[i].parameters())

    def __repr__(self):
        repr = '\n'.join(str(layer) for layer in self.layers)
        return f"MLP of [{repr}]"
