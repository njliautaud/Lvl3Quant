import json
d=json.load(open('/home/jupiter/dow_ic_results.json'))
for k,v in d.items():
    print(k + ' IC=' + str(v['ic']) + ' ndays=' + str(v['n_days']))
