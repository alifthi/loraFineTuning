import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig,PeftModel
from trl import SFTTrainer
from config import (MODEL_NAME, LORA_R, LORA_ALPHA,  
                    LORA_DROPOUT, TASK_TYPE, OUTPUTS_DIR, 
                    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, OPTIMIZER,
                    LR_SCHEDULAR, SAVE_STEPS, MAX_SEQ_LENGTH, MODEL_PATH)
class Model:
    def __init__(self,tokenizer,trainDataset,name) -> None:
        self.name=name
        self.trainDataset=trainDataset
        self.tokenizer=tokenizer
        self.loadModel()
        self.setConfigs()
    def train(self):
        self.trainer.train()
    def setConfigs(self):
        self.loraConfig=LoraConfig(
           lora_alpha=LORA_ALPHA,
           r=LORA_R,
            lora_dropout=LORA_DROPOUT,
            task_type=TASK_TYPE,
            bias='all',
            target_modules=["q_proj", "k_proj"])
        self.trainingAguments=TrainingArguments(
            output_dir=OUTPUTS_DIR[:-1]+'_'+self.name+'/',
            num_train_epochs=EPOCHS,
            optim=OPTIMIZER,
            logging_steps=16,
            save_steps=SAVE_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
           lr_scheduler_type=LR_SCHEDULAR,
            report_to="tensorboard"
        )
        self.trainer=SFTTrainer(
            model=self.model,
            train_dataset=self.trainDataset['train'],
            peft_config=self.loraConfig,
            dataset_text_field='text',
            max_seq_length=MAX_SEQ_LENGTH,
            tokenizer=self.tokenizer,
            args=self.trainingAguments            
        )
    def inference(self,modelChkpnt):
        device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.model=PeftModel.from_pretrained(self.model,modelChkpnt,
                                             torch_dtype=torch.float16,
                                             device=device)
        messages=[]
        while True:
            prompt=input("prompt: ")
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,num_beams=8,
                num_return_sequences=1,top_k=200,
                top_p=0.9,temperature=0.9)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            messages.append({"role": "assistant", "content": response})
            print(response) 
    def loadModel(self):
        self.model=AutoModelForCausalLM.from_pretrained(MODEL_NAME,device_map='auto')
    def saveModel(self):
        self.model.save_pretrained(MODEL_PATH+self.modelName+'_'+self.name+'/')