import threading
from main import main
from config import NUM_TRAINING_STEP
r1=main('Agent1')
r2=main('Agent2')
for _ in range(NUM_TRAINING_STEP):
    for w1,w2 in zip(r1.model.model.parameters(),r2.model.model.parameters()):
        aggrigate=w1+w2
        aggrigate=aggrigate/2
        w1.data.copy_(aggrigate)
        w2.data.copy_(aggrigate)
    t1=threading.Thread(target=r1.init)
    t2=threading.Thread(target=r2.init)

    t1.start()
    t2.start()

    t1.join()
    t2.join()