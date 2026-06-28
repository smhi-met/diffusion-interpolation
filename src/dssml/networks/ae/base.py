from abc import ABC, abstractmethod

class IAutoEncoder(ABC):
    """
    Base class for AutoEncoders. Subclasses must implement:
      - encode(x) -> z
      - decode(z) -> x_recon
    """
 
    @abstractmethod
    def encode(self, x):
        ...
    @abstractmethod
    def decode(self, z):
        ...

 