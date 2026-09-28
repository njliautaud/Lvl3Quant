import subprocess, os, sys
ray_bin = "/home/jupiter/miniconda3/envs/ray311/bin/ray"
proc = subprocess.Popen(
    [ray_bin, "start", "--head", "--port=6379",
     "--node-ip-address=jupiter",
     "--dashboard-host=0.0.0.0",
     "--dashboard-port=8265"],
    stdout=open("/home/jupiter/ray_head.log", "w"),
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True
)
print("PID:", proc.pid)
