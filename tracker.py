import psutil
import time
import sys
import os
import signal

start = time.time()

def check_pid(pid):
    if psutil.pid_exists(pid):
        return True
    else:
        return False

PID = int(sys.argv[1])

iteration_num = 0

mean_rss = mean_vms = 0
max_rss = max_vms = 0

while (check_pid(PID)):
    iteration_num += 1
    process = psutil.Process(PID)
    current_rss = process.memory_info().rss
    current_vms = process.memory_info().vms
    
    mean_rss += current_rss
    mean_vms += current_vms
    
    max_rss = max(max_rss, current_rss)
    max_vms = max(max_vms, current_vms)

    time.sleep(5)

process_time = time.time() - start

mean_rss /= iteration_num * 2**30
max_rss /= 2**30

mean_vms /= iteration_num * 2**30
max_vms /= 2**30

os.system("telegram-send 'PID: {}; TIME: {}; MEAN: {}r, {}v; MAX: {}r, {}v'".format(
    PID, process_time, mean_rss, mean_vms, max_rss, max_vms
))
