from datasets import load_from_disk
from transformers import AutoTokenizer
class utils:
    def __init__(self, modelName='Qwen/Qwen1.5-0.5B-Chat', dataDir='../data/alpaca') -> None:
        self.dataDir=dataDir
        self.modelName=modelName
        self.data=self.loadData()
    def loadData(self):
        return load_from_disk(self.dataDir)
    def preprocessing(self):
        self.tokenizer=AutoTokenizer.from_pretrained(self.modelName)
        