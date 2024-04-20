import threading
from main import main

r1=main('Agent1')
r2=main('Agent2')
t1=threading.Thread(target=r1.init)
t2=threading.Thread(target=r2.init)
t1.start()
t2.start()
t1.join()
t2.join()