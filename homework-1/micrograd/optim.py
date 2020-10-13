class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Applying gradient descent to parameters"""
        for i, parameter in enumerate(self.parameters):
            # print(f'before {i}', self.parameters[i])
            self.parameters[i].data -= parameter.grad * self.lr
            # print(f'after {i}', self.parameters[i])
        # Implement SGD!

    def zero_grad(self):
        """Resetting gradient for all parameters (set gradient to zero)"""
        for i, parameter in enumerate(self.parameters):
            self.parameters[i].grad = 0
