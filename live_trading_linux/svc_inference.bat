@echo off
set PYTHONPATH=C:\Users\claude\Lvl3Quant
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
set PYTHONUNBUFFERED=1
cd /d C:\Users\claude\Lvl3Quant\live_trading
C:\Python311\python.exe paper_trading_mamba.py --symbol NQM6 --exchange CME --device cuda --min-tier Top1%% --window 1000 --stride 500 --follow-events
