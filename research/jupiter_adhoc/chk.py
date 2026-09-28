import os
lines=open('/home/jupiter/sync_derived_uranus.log').readlines()
ok=sum(1 for l in lines if 'OK' in l)
fail=sum(1 for l in lines if 'FAIL' in l)
print('lines:',len(lines),'OK:',ok,'FAIL:',fail)
for l in lines[-3:]: print(l.rstrip())
