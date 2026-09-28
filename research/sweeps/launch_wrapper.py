"""Launcher that creates a truly detached process on Windows with internal logging."""
import subprocess, sys, os

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

# The child script will handle its own logging via the --log-file arg
p = subprocess.Popen(
    [sys.executable, '-u', '-c',
     'import sys, os; '
     'os.chdir(r"C:\\Users\\claude\\Lvl3Quant"); '
     'sys.stdout = open("output/confluence_meta_v4/train_stdout.log", "w", buffering=1); '
     'sys.stderr = open("output/confluence_meta_v4/train_stderr.log", "w", buffering=1); '
     'sys.argv = ["train", "--out-dir", "output/confluence_meta_v4"]; '
     'exec(open("alpha_discovery/deep_models/train_confluence_meta_v4.py").read())'
    ],
    cwd=r'C:\Users\claude\Lvl3Quant',
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
    close_fds=True,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print(f'PID={p.pid}')
