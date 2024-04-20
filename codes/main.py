import os
os.environ["OMP_NUM_THREADS"] = "4"
from utils.utils import utils
from model.Model import Model
utils=utils()
model=Model(tokenizer=utils.tokenizer, trainDataset=utils.data)
model.train()
model.saveModel()