import numpy as np
from typing import Union


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __pow__(self, other: Union[int, float]) -> "Value":
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), "pow")

        def _backward():
            self.grad += out.grad * (self.data ** (other - 1) * other)

        out._backward = _backward

        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.grad * np.exp(self.data)

        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad * 1 / self.data

        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.max([0., self.data]), (self,), "relu")

        def _backward():
            self.grad += 0 if self.data < 0 else out.grad

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

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __le__(self, other):
        if isinstance(other, Value):
            return self.data <= other.data
        return self.data <= other

    def __lt__(self, other):
        if isinstance(other, Value):
            return self.data < other.data
        return self.data < other

    def __gt__(self, other):
        if isinstance(other, Value):
            return self.data > other.data
        return self.data > other

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class Tensor:
    """
    Tensor is a kinda array with expanded functianality.

    Tensor is very convenient when it comes to matrix multiplication,
    for example in Linear layers.
    """

    def __init__(self, data):
        # print('data', data)
        self.data = np.array(data, dtype=object)
        if len(data.shape) == 2:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if not isinstance(self.data[i][j], Value):
                        self.data[i][j] = Value(self.data[i][j])
        else:
            for i in range(data.shape[0]):
                if not isinstance(self.data[i], Value):
                    self.data[i] = Value(self.data[i])

    def __add__(self, other):
        if isinstance(other, Tensor):
            if self.shape() == other.shape():
                return Tensor(np.add(self.data, other.data))
            elif len(self.shape()) == 2 and len(other.shape()) == 1 and self.shape()[1] == other.shape()[0]:
                return Tensor(np.add(self.data, other.data))
            else:
                assert 1 == 2, 'problems with shape'
        return Tensor(self.data + other)

    def __mul__(self, other):
        return Tensor(self.data * other.data) # ...

    def __truediv__(self, other):
        return self.data / other.data  # ...

    def __floordiv__(self, other):
        # 0 problem
        return self.data // other.data  # ...

    def __rtruediv__(self, other):  # other / self
        return Tensor(other / self.data)

    def __radd__(self, other):
        if isinstance(other, float):
            return Tensor(self.data + other)
        return other.data + self.data  # ...

    def __rmul__(self, other):
        return other.data * self.data  # ...

    def exp(self):
        return Tensor(np.exp(self.data))  # ...

    def dot(self, other):
        if isinstance(other, Tensor):
            return Tensor(np.dot(self.data, other.data)) #...
        return self.data * other  # ...

    def sum(self, dim):
        return Tensor(np.sum(self.data, axis=dim))

    def mean(self, dim):
        return Tensor(np.mean(self.data, axis=dim))

    def shape(self):
        return self.data.shape

    def argmax(self, dim=None):
        return np.argmax(self.data, axis=dim)  # ...

    def max(self, dim=None):
        return np.max(self.data, axis=dim)  # ...

    def relu(self):
        return Tensor(np.vectorize(lambda x: x.relu())(self.data))

    def log(self):
        return Tensor(np.vectorize(lambda x: x.log())(self.data))

    def __neg__(self):  # -self
        return Tensor(self.data * -1)

    # def __rsub__(self, other):  # other - self
    #     assert isinstance(other, float)
    #     return Tensor(other - self.data)

    def reshape(self, *args, **kwargs):
        self.data = np.reshape(self.data, *args, **kwargs)  # ...
        return self

    def backward(self):
        for value in self.data.flatten():
            value.backward()

    def parameters(self):
        return list(self.data.flatten())

    def __repr__(self):
        return "Tensor\n" + str(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def item(self):
        return self.data.flatten()[0].data
