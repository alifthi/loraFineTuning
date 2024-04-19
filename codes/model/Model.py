import torch
from tansformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel
from config import MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT, TASK_TYPE
class Model:
    def __init__(self) -> None:
        self.trainingAguments=None
        self.trainer=None
        self.setLoraConfig()
        self.model=self.loadModel()
    def train(self):
        pass
    def inference(self):
        pass
    def loadModel(self):
        return AutoModelForCausalLM.from_pretrained(MODEL_NAME,device_map='auto')
    def saveModel(self):
        pass
    def quantizationConfig(self):
        pass
    def setLoraConfig(self):
        self.loraConfig=LoraConfig(
            lora_alpha=LORA_ALPHA,
            r=LORA_R,
            lora_dropout=LORA_DROPOUT,
            task_type=TASK_TYPE,
            bias='none'
        )