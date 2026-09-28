cnn_by_date_compact = {}
for d,v in cnn_by_date.items():
    cnn_by_date_compact[d.replace("-","")] = v
cnn_by_date = cnn_by_date_compact
