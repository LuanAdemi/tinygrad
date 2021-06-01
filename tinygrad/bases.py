from tinygrad.tensor import Tensor
import pickle
import tqdm

## base classes for basic module creation
class Base:
    def __init__(self):
        super().__init__()
    
    @property
    def base(self):
        return NotImplementedError

class Layer(Base):
    def __init__(self):
        super().__init__()
        
    def __call__(self, x):
        self.forward(x)
    
    def forward(self, x):
        raise NotImplementedError

    @property
    def base(self):
        return "Layer"

class Activation(Base):
    def __init__(self):
        super().__init__()
        
    def __call__(self, x):
        self.forward(self, x)
        
    def forward(self, x):
        raise NotImplementedError
    
    @property
    def base(self):
        return "Activation"

class Loss(Base):
    def __init__(self):
        super().__init__()
        
    def __call__(self, gt, out):
        return self.criterion(gt, out)
    
    def criterion(self, gt, out):
        raise NotImplementedError
    
    @property
    def base(self):
        return "Loss"

class Module(Base):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return NotImplementedError
    
    def state_dict(self):
        return self.__dict__
    
    # TODO: find a more general approach.
    # This works for now though
    @property
    def parameters(self):
        parameters = []
        attributes = list(self.state_dict().values())
        for attr in attributes:
            if attr.base == "Layer":
                paras = attr.parameters
                for para in paras:
                    parameters.append(para)
        return parameters
            
    # saves the parameters
    def save_parameters(self, file, info=True):
        try:
            outfile = open(file, 'wb')
        except:
            raise FileNotFoundError(f"{file}: no such file or directory")
        
        pickle.dump(self.state_dict(), outfile)
        
        if info:
            print(f"[Info] Saved the state_dict to {file}")

    # loads the parameters
    def load_parameters(self, file, info=True):
        try:
            infile = open(file, 'rb')
        except:
            raise FileNotFoundError("f{file}: no such file or directory")
        
        state_dict = pickle.load(infile)
        
        try:
            for key in state_dict.keys():
                self.__dict__[key] = state_dict[key]
            if info:
                print("[Info] All keys matched successfully!")
        except:
            raise "The state_dict does not match the module keys"


