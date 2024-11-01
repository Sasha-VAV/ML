import random
import numpy as np
import math


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data=None, _children=(), _op=''):
        if data is None:
            data = random.random()*2 - 1
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None # function
        self._prev = set(_children) # set of Value objects
        self._op = _op # the op that produced this node, string ('+', '-', ....)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), '**')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad if self.data else 0
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'r')

        def _backward():
            self.grad += 0 if self.data <= 0 else out.grad
        out._backward = _backward

        return out

    def softmax(self, other_values):
        input = other_values.append(self.data)
        input.sort()
        ii = input.index(self.data)
        sum_j_k = sum(math.exp(input[j]) for j in range(len(input)))
        probability = np.array([math.exp(input[i]) / sum_j_k for i in range(len(input))])
        out = Value(probability[ii], (self,), 'softmax')

        def _backward():
            delta = 1e-3
            new_input = tuple(input[i] + delta for i in range(len(input)))
            new_sum_j_k = sum(math.exp(new_input[i]) for i in range(len(input)))
            new_probability = np.array([math.exp(new_input[i]) / new_sum_j_k for i in range(len(input))])
            res = list((new_probability[i] - probability[i]) / delta for i in range(len(input)))
            for i in range(len(input)):
                if i==ii:
                    self.grad += res[i] * out.grad
                else:
                    other_values[i].grad += res[i] * out.grad


        out._backward = _backward
        return out


    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
