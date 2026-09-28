import json,sys
d=json.load(open('/home/jupiter/dow_ic_results.json'))
msg=str(d)
raise RuntimeError(msg)
