import lightning as L
from abc import ABC, abstractmethod
from torch.utils.data import  Dataset

class DataModuleBase(L.LightningDataModule, ABC):
   @property
   @abstractmethod
   def name(self) -> str:
      ...
   
 
   @property
   @abstractmethod
   def manifest(self) -> str:
      ...
   
 
   @property
   @abstractmethod
   def val_dataset(self) -> Dataset:
      ...
 
   @property
   @abstractmethod
   def train_dataset(self) -> Dataset:
      ...
   
   @property
   @abstractmethod
   def test_dataset(self) -> Dataset:
      ...
      