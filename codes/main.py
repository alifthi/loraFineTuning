import os
os.environ["OMP_NUM_THREADS"] = "4"
from utils.utils import utils
from model.Model import Model
class main:
    def __init__(self,name) -> None:    
        self.utils=utils(name)
        self.model=Model(tokenizer=self.utils.tokenizer, trainDataset=self.utils.data, name=name)
    def init(self):
        self.model.train()
        self.model.saveModel()