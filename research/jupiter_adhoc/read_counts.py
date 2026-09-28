import json
d=json.load(open('/home/jupiter/midday_signal_counts.json'))
for k,v in d.items():
    if k != 'day_stats':
        print(k + ':' + str(v))
