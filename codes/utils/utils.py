from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, DATA_DIR
class utils:
    def __init__(self) -> None:
        self.dataDir=DATA_DIR
        self.modelName=MODEL_NAME
        self.data=self.loadData()
    def loadData(self,dataset=None):
        return load_from_disk(self.dataDir) if isinstance(dataset,type(None)) else load_dataset(dataset)
    def preprocessing(self):
        self.tokenizer=AutoTokenizer.from_pretrained(self.modelName)
        