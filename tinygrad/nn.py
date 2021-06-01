from tinygrad.tensor import Tensor
import pickle
from tqdm import tqdm
from tinygrad.bases import Layer, Loss, Activation

# children classes
class Linear(Layer):
    def __init__(self, inp, out, bias=False):
        self.W = Tensor.uniform(inp, out)
        self.B = None
        
        if bias:
            self.B = Tensor.uniform(out)

    def __call__(self, x):
        if self.B is not None:
            return x.dot(self.W).add(self.B)
        else:
            return x.dot(self.W)
    
    @property
    def parameters(self):
        if self.B is not None:
            return [self.W, self.B]
        else:
            return [self.W]

class RMSE(Loss):
    def __init__(self):
        super().__init__()
    
    def criterion(self, gt, out):
        error = out.sub(gt)
        squared_error = error.mul(error)
        return squared_error.mean()
        
class Trainer:
    """
    A class that trains a given model with the specified optimizer and 
    loss function
    """
    def __init__(self, model, optimizer, loss_fn=RMSE()):
        self.model = model
        self.optimizer = optimizer
        
        self.loss_fn = loss_fn
        
        self.loss_history = []

    def train(self, X, Y, epochs=100):
        with tqdm(range(epochs), unit="batch") as tepochs:
            for epoch in tepochs:
                tepochs.set_description(f"Epoch {epoch}")
                
                out = self.model(X)

                loss = self.loss_fn(out, Y)
                
                loss.backward()
                self.optimizer.step()
                
                self.loss_history.append(loss.item())
                tepochs.set_postfix(loss=loss.item())
        
        return self.loss_history
                
class ReLU(Activation):
    def __init__(self, leaky=False):
        self.leaky = False
    
    def __call__(self, x):
        if self.leaky:
            return x.leakyrelu()
        else:
            return x.relu()

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
    def __call__(self, x):
        return x.sigmoid()

class Dropout:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return x.dropout(p=self.p)

class BatchNorm2D:
  def __init__(self, sz, eps=1e-5, track_running_stats=False, training=False, momentum=0.1):
    self.eps, self.track_running_stats, self.training, self.momentum = eps, track_running_stats, training, momentum

    self.weight, self.bias = Tensor.ones(sz), Tensor.zeros(sz)

    self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)

  def __call__(self, x):
    if self.track_running_stats or self.training:
      batch_mean = x.mean(axis=(0,2,3))
      y = (x - batch_mean.reshape(shape=[1, -1, 1, 1]))
      batch_var = (y*y).mean(axis=(0,2,3))

    if self.track_running_stats:
      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
      if self.num_batches_tracked is None: self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)
      self.num_batches_tracked += 1

    if self.training:
      return self.normalize(x, batch_mean, batch_var)

    return self.normalize(x, self.running_mean, self.running_var)

  def normalize(self, x, mean, var):
    x = (x - mean.reshape(shape=[1, -1, 1, 1])) * self.weight.reshape(shape=[1, -1, 1, 1])
    return x.div(var.add(self.eps).reshape(shape=[1, -1, 1, 1])**0.5) + self.bias.reshape(shape=[1, -1, 1, 1])

