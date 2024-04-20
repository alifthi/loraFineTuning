from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, DATA_DIR
class utils:
    def __init__(self) -> None:
        self.dataDir=DATA_DIR
        self.modelName=MODEL_NAME
        self.data=self.loadData()
        self.preprocessing()
    def loadData(self,dataset=None):
        return load_from_disk(self.dataDir) if isinstance(dataset,type(None)) else load_dataset(dataset,split='train')
    def preprocessing(self):
        self.tokenizer=AutoTokenizer.from_pretrained(self.modelName)
        self.data=self.data.map(self.convertToChatTemplate)
    def convertToChatTemplate(self,text):
        text=text['text'].split('### Human:')[-1]
        text=text.split('### Assistant:')
        userContent=text[0]
        assistantContent=text[1]
        chat=[{'role':'user','content':userContent},
              {'role':'assistant','content':assistantContent}]
        chat=self.tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)
        return {'text':chat}
