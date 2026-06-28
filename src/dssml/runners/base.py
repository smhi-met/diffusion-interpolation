from abc import ABC, abstractmethod

class RunnerBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...
    
    @abstractmethod
    def run(self):
        ...