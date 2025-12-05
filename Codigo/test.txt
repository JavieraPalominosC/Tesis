import os, sys

# function to monitor RAM and CPU usage
def monitor_psrecord(psrecord_name):
    # monitor CPU and memory usage                                                         
    pid = os.getpid()
    L = ['psrecord', "%s" % pid, "--interval", "1", "--log", "psrecord/psrecord_log_%s_%s.txt" % (psrecord_name, pid)]
    os.spawnvpe(os.P_NOWAIT, 'psrecord', L, os.environ)

# launch psrecord                                                               
monitor_psrecord("test_%s_%s" % (os.getenv("SLURM_JOB_ID"), os.getenv("SLURM_ARRAY_TASK_ID")))

# start
lines = [i.strip() for i in open("test.txt").readlines()]
nline = int(sys.argv[1]) - 1
print("Testing %s" % lines[nline].strip())

# use ~500 MB of memory                                                           
x = bytearray(1024*1024*500)

# use 100% CPU for some seconds                                                                          
y0 = 1.01
y = y0
for i in range(200000000):
    y = y * y0

print("Done from python. Job ID: %s, array task ID: %s" % (os.getenv("SLURM_JOB_ID"), os.getenv("SLURM_ARRAY_TASK_ID")))