from tinygrad.tensor import Tensor
from tinygrad.base import Activation, Loss

## define some common loss functions
class RMSE(Loss):
    def __init__(self, root=True):
        super().__init__()

    def forward(self, gt, out):
        squared_mean_error = out.sub(gt).pow(2).mean()
        if root:
            return squared_error.sqrt()
        return squared_error
            
## define some common activation functions
class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.sigmoid()

class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.tanh()

class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.relu()

class Softmax(Activation):
    def __init__(self):
       super().__init__()

    def forward(self, x):
        return x.softmax()


