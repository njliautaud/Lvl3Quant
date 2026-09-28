import subprocess, os
ray_bin = "/home/jupiter/miniconda3/envs/ray311/bin/ray"
proc = subprocess.Popen([ray_bin, "start", "--head", "--port=6379", "--node-ip-address=jupiter", "--dashboard-host=0.0.0.0", "--dashboard-port=8265"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out, err = proc.communicate(timeout=25)
print("OUT:", out)
print("ERR:", err)
print("RC:", proc.returncode)
