@echo off
REM Background launcher for Windows Task Scheduler / startup
REM Runs live_stack.py detached — logs to live_trading/logs/

cd /d C:\Users\claude\Lvl3Quant\live_trading
set PYTHONPATH=C:\Users\claude\Lvl3Quant
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
set PYTHONUNBUFFERED=1

start /B "" C:\Python311\python.exe live_stack.py --symbol ESM6 --exchange CME --device cuda --min-tier "Top1%%" > logs\live_stack_bg.log 2>&1
