#!/usr/bin/env python3
"""
Monitor the deep benchmark progress and send Discord updates.
"""
import time
import os
import sys
import subprocess
import re
import psutil
from pathlib import Path
from datetime import datetime

LOG_FILE = Path(__file__).parent / 'results' / 'deep_benchmark_live.log'
DEEP_PID = 42136  # The run_deep_after_arch.py process

def send_discord(message):
    """Send message to Discord using Node.js discord helper."""
    try:
        bridge_dir = Path("C:/Users/Footb/Documents/Github/teleclaude-main")
        script = f"""
const {{ sendToDiscord }} = require('./lib/discord');
sendToDiscord({repr(message)}).then(() => process.exit(0)).catch(e => {{ console.error(e); process.exit(1); }});
"""
        result = subprocess.run(
            ['node', '-e', script],
            cwd=str(bridge_dir),
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode != 0:
            print(f"Discord error: {result.stderr}")
    except Exception as e:
        print(f"Failed to send Discord: {e}")

def get_process_status(pid):
    try:
        proc = psutil.Process(pid)
        mem_gb = proc.memory_info().rss / (1024**3)
        children = proc.children(recursive=True)
        child_mem = 0
        for c in children:
            try:
                child_mem += c.memory_info().rss / (1024**3)
            except:
                pass
        return proc.status(), mem_gb + child_mem
    except psutil.NoSuchProcess:
        return 'dead', 0

def read_log_tail(n=100):
    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        return lines[-n:]
    except:
        return []

def parse_progress(lines):
    """Extract key info from recent log lines."""
    # Join lines into text for parsing
    text = ''.join(lines)
    
    info = {
        'current_arch': None,
        'current_phase': None,
        'current_fold': None,
        'ic_results': [],
        'completed': [],
        'errors': [],
    }
    
    # Find current arch being tested
    for line in reversed(lines):
        if 'TESTING:' in line:
            m = re.search(r'TESTING: (\w+)', line)
            if m:
                info['current_arch'] = m.group(1)
                break
    
    # Find current phase
    for line in reversed(lines):
        if 'PHASE' in line and ':' in line:
            m = re.search(r'PHASE (\d+):', line)
            if m:
                info['current_phase'] = m.group(1)
                break
    
    # Find current fold
    for line in reversed(lines):
        if 'Fold' in line and 'IC=' in line:
            m = re.search(r'Fold (\d+).*IC=([\d.]+)', line)
            if m:
                info['current_fold'] = (int(m.group(1)), float(m.group(2)))
                break
    
    # Find completed results
    for line in lines:
        if 'IC=' in line and ('RESULTS' in line or 'COMPLETED' in line or ('IC=' in line and 'HR=' in line)):
            ic_m = re.search(r'IC=([\d.]+)', line)
            hr_m = re.search(r'HR=([\d.]+)%', line)
            if ic_m:
                info['ic_results'].append({
                    'ic': float(ic_m.group(1)),
                    'hr': float(hr_m.group(1)) if hr_m else None,
                    'line': line.strip()[-100:]
                })
    
    # Find COMPLETED markers
    for line in lines:
        if 'COMPLETED:' in line and 'day' in line.lower():
            info['completed'].append(line.strip())
    
    # Find errors
    for line in lines[-30:]:
        if 'ERROR' in line.upper() or 'FAILED' in line.upper() or 'OOM' in line.upper() or 'CUDA out of memory' in line:
            info['errors'].append(line.strip()[-150:])
    
    return info

def main():
    print(f"Deep benchmark monitor started at {datetime.now()}")
    print(f"Monitoring PID {DEEP_PID}, log: {LOG_FILE}")
    
    last_log_size = 0
    last_update_time = time.time()
    update_interval = 300  # 5 minutes
    last_completed_count = 0
    check_count = 0
    
    while True:
        check_count += 1
        status, mem_gb = get_process_status(DEEP_PID)
        
        if status == 'dead':
            # Check if it completed normally
            lines = read_log_tail(50)
            final_text = ''.join(lines)
            if 'DEEP LEARNING BENCHMARK COMPLETE' in final_text:
                send_discord("**Deep Benchmark COMPLETE!** All phases finished. Check the log for final results.")
            else:
                send_discord(f"**WARNING:** Deep benchmark process (PID {DEEP_PID}) appears to have died. Check logs.")
            print(f"Process {DEEP_PID} is dead, exiting monitor.")
            break
        
        # Read current log
        try:
            log_size = os.path.getsize(LOG_FILE)
        except:
            log_size = 0
        
        log_changed = log_size != last_log_size
        last_log_size = log_size
        
        lines = read_log_tail(80)
        info = parse_progress(lines)
        
        now = time.time()
        time_since_update = now - last_update_time
        
        # Send update if:
        # 1. New completions detected
        # 2. Time-based update (every 5 min)
        new_completions = len(info['completed']) > last_completed_count
        
        if new_completions or time_since_update >= update_interval:
            last_update_time = now
            last_completed_count = len(info['completed'])
            
            # Build status message
            phase = info['current_phase'] or '?'
            arch = info['current_arch'] or 'unknown'
            fold_info = f"Fold {info['current_fold'][0]}, IC={info['current_fold'][1]:.4f}" if info['current_fold'] else "loading/preparing"
            
            msg_parts = [
                f"**Deep Benchmark Update** ({datetime.now().strftime('%H:%M')})",
                f"Phase {phase} | Currently: {arch}",
                f"Latest fold: {fold_info}",
                f"Memory: {mem_gb:.1f}GB",
            ]
            
            if info['completed']:
                msg_parts.append(f"\nCompleted: {len(info['completed'])} stages")
                for c in info['completed'][-3:]:
                    msg_parts.append(f"  - {c[-100:]}")
            
            if info['errors']:
                msg_parts.append(f"\nERRORS detected:")
                for e in info['errors'][-2:]:
                    msg_parts.append(f"  ! {e}")
            
            # Look for IC results in recent lines
            for line in lines[-30:]:
                if 'IC=' in line and 'ICIR=' in line:
                    msg_parts.append(f"\nResult: {line.strip()[-120:]}")
                    break
            
            message = '\n'.join(msg_parts)
            send_discord(message)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent update: phase={phase}, arch={arch}")
        
        # Check every 60 seconds
        time.sleep(60)

if __name__ == '__main__':
    main()
