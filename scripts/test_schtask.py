import time
f = open(r'C:\Users\claude\Lvl3Quant\output\schtask_alive.txt', 'w')
f.write('STARTED\n'); f.flush()
time.sleep(120)
f.write('SURVIVED 2MIN\n'); f.close()
