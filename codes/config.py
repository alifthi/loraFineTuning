MODEL_NAME='Qwen/Qwen1.5-0.5B-Chat'
DATA_DIR='../data/alpaca'
MODEL_PATH='../TrainedModels/'
'''LoRA configs'''
LORA_R=64
LORA_ALPHA=16
LORA_DROPOUT=0.1
TASK_TYPE='CAUSAL_LM'

'''Training Parametrs'''
OUTPUTS_DIR='../outputs/'
EPOCHS=1
LEARNING_RATE=0.001
WEIGHT_DECAY=0.001
OPTIMIZER='adamw_torch'
LR_SCHEDULAR='cosine'
SAVE_STEPS=100
MAX_SEQ_LENGTH=128

'''Federated Learning Parameters'''
NUM_TRAINING_STEP=5