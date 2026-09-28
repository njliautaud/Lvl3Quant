
import json
d = json.load(open('/home/jupiter/iceberg_highcv_aggregate.json'))
print('BEST:', json.dumps(d[0]))
print('SECOND:', json.dumps(d[1]))
print('Total configs:', len(d))
pos = [r for r in d if r['sortino'] > 0]
print('Positive Sortino:', len(pos))
cv_summary = {}
for r in d:
    cv = r['cv_tier']
    if cv not in cv_summary:
        cv_summary[cv] = {'pos': 0, 'total': 0, 'best_sortino': -999}
    cv_summary[cv]['total'] += 1
    if r['sortino'] > 0:
        cv_summary[cv]['pos'] += 1
    if r['sortino'] > cv_summary[cv]['best_sortino']:
        cv_summary[cv]['best_sortino'] = r['sortino']
for cv, s in sorted(cv_summary.items()):
    print(cv, s)
