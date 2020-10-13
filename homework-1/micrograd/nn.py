import numpy as np

from engine import Value, Tensor


class Module:
    """
    Base class for every layer.
    """
    def forward(self, *args, **kwargs):
        """Depends on functionality"""
        pass

    def __call__(self, *args, **kwargs):
        """For convenience we can use model(inp) to call forward pass"""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Return list of trainable parameters"""
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        """Initializing model"""
        stvd = 1. / np.sqrt(in_features)
        self.W = Tensor(np.random.uniform(-stvd, stvd, size=(in_features, out_features)))
        self.bias = bias
        if bias:
            self.b = Tensor(np.random.uniform(-stvd, stvd, size=out_features))
        # Create Linear Module

    def forward(self, inp):
        """Y = W * x + b"""
        out = inp.dot(self.W)  # .dot(self.W.data.transpose((1, 0)))
        if self.bias:
            out += self.b
        return out

    def parameters(self):
        parameters_list = self.W.parameters()
        if self.bias:
            parameters_list.extend(self.b.parameters())
        return parameters_list


class ReLU(Module):
    """The most simple and popular activation function"""
    def forward(self, inp):
        # Create ReLU Module
        return inp.relu()  # ...


class Sigmoid(Module):

    def forward(self, inp):
        return 1. / (1. + (-inp).exp())


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification"""
    def forward(self, prediction, targets):
        return -((-prediction + 1).log() * (- targets + 1) + (prediction.log() * targets)).mean(0)
        # Create CrossEntropy Loss Module
        # return -(inp.log() * label).sum(dim=0)  #  .mean(0) ...
