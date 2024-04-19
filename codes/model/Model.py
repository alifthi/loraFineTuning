import torch
from tansformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from config import (MODEL_NAME, LORA_R, LORA_ALPHA,  
                    LORA_DROPOUT, TASK_TYPE, OUTPUTS_DIR, 
                    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, OPTIMIZER,
                    LR_SCHEDULAR, SAVE_STEPS, MAX_SEQ_LENGTH)
class Model:
    def __init__(self,tokenizer,trainDataset) -> None:
        self.trainDataset=trainDataset
        self.tokenizer=tokenizer
        self.setConfigs()
        self.model=self.loadModel()
    def train(self):
        self.trainer.train()
    def inference(self):
        pass
    def loadModel(self):
        return AutoModelForCausalLM.from_pretrained(MODEL_NAME,device_map='auto')
    def saveModel(self):
        pass
    def quantizationConfig(self):
        pass
    def setConfigs(self):
        self.loraConfig=LoraConfig(
            lora_alpha=LORA_ALPHA,
            r=LORA_R,
            lora_dropout=LORA_DROPOUT,
            task_type=TASK_TYPE,
            bias='none')
        self.trainingAguments=TrainingArguments(
            output_dir=OUTPUTS_DIR,
            num_train_epochs=EPOCHS,
            optim=OPTIMIZER,
            save_steps=SAVE_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULAR,
            report_to="tensorboard"
        )
        self.trainer=SFTTrainer(
            model=self.model,
            train_dataset=self.trainDataset,
            peft_config=self.loraConfig,
            dataset_text_field='text',
            max_seq_length=MAX_SEQ_LENGTH,
            tokenizer=self.tokenizer,
            args=self.trainingAguments            
        )