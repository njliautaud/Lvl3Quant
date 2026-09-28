import re

total_h900 = 0
total_h600 = 0
current_formula = None
results = {}
count = 0

with open(r'C:\Users\Footb\Documents\Github\Lvl3Quant\alpha_discovery\results\ga_fillsim_run.log', 'rb') as f:
    for raw_line in f:
        line = raw_line.decode('utf-8', errors='ignore').strip()
        if '--- Formula:' in line:
            if current_formula:
                results[current_formula] = (total_h900, total_h600, count)
            m = re.search(r'Formula: (\S+)', line)
            if m:
                current_formula = m.group(1)
            total_h900 = 0
            total_h600 = 0
            count = 0
            continue
        # Match dollar amounts like $-1234 or $567
        m900 = re.search(r't3\.5_h900s:\$([+-]?\d[\d,]*)', line)
        m600 = re.search(r't3\.5_h600s:\$([+-]?\d[\d,]*)', line)
        if m900:
            val = int(m900.group(1).replace(',', ''))
            total_h900 += val
            count += 1
        if m600:
            val = int(m600.group(1).replace(',', ''))
            total_h600 += val

if current_formula:
    results[current_formula] = (total_h900, total_h600, count)

for name, (h900, h600, cnt) in results.items():
    print(f'{name} ({cnt} days): h900s=${h900:+,} | h600s=${h600:+,}')
