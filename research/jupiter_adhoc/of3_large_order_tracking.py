import sys, logging, numpy as np, databento as db
from pathlib import Path
logging.basicConfig(format="%(asctime)s [of3] %(message)s", level=logging.INFO)
log = logging.getLogger("of3")
LARGE_THRESH = 50
RTH_START_S = 52200
RTH_END_S = 75600
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/of3_large_order")
OUT_DIR.mkdir(parents=True, exist_ok=True)
def process_day(mbo_path):
    date_str = mbo_path.stem.split("-")[-1]
    out_path = OUT_DIR / f"{date_str}_of3.npz"
    if out_path.exists(): log.info(f"skip {date_str}"); return True
    log.info(f"processing {date_str}")
    store = db.DBNStore.from_file(str(mbo_path))
    df = store.to_df().reset_index()
    ts = df["ts_event"].dt.tz_convert("UTC")
    base = ts.dt.normalize()
    rth_s = base + np.timedelta64(RTH_START_S,"s")
    rth_e = base + np.timedelta64(RTH_END_S,"s")
    df = df[(ts >= rth_s) & (ts < rth_e)].reset_index(drop=True)
    if len(df) == 0: log.warning(f"no RTH data {date_str}"); return False
    n = len(df)
    acts = df["action"].values
    sizes = df["size"].values.astype(np.float32)
    sides = df["side"].values
    oids = df["order_id"].values
    is_large = sizes >= LARGE_THRESH
    is_add = acts == "A"
    is_cancel = acts == "C"
    is_trade = acts == "T"
    large_add_oids = set(oids[is_large & is_add])
    trade_oid_counts = {}
    for o in oids[is_trade]: trade_oid_counts[o] = trade_oid_counts.get(o,0)+1
    iceberg_oids = set(o for o,c in trade_oid_counts.items() if c > 1)
    large_nearby = np.zeros(n, dtype=np.float32)
    iceberg_active = np.zeros(n, dtype=np.float32)
    large_imbalance = np.zeros(n, dtype=np.float32)
    cancel_rate = np.zeros(n, dtype=np.float32)
    WIN = 500
    add_count = 0; cancel_count = 0
    bid_large = 0; ask_large = 0; large_in_win = 0
    from collections import deque
    win_acts = deque(); win_is_large = deque(); win_sides = deque(); win_oids = deque()
    for i in range(n):
        a=acts[i]; s=sizes[i]; sd=sides[i]; o=oids[i]
        win_acts.append(a); win_is_large.append(is_large[i]); win_sides.append(sd); win_oids.append(o)
        if a=="A": add_count+=1
        if a=="C": cancel_count+=1
        if is_large[i] and a=="A":
            large_in_win+=1
            if sd=="B": bid_large+=1
            elif sd=="A": ask_large+=1
        if len(win_acts) > WIN:
            old_a=win_acts.popleft(); old_l=win_is_large.popleft(); old_sd=win_sides.popleft(); win_oids.popleft()
            if old_a=="A": add_count-=1
            if old_a=="C": cancel_count-=1
            if old_l and old_a=="A":
                large_in_win-=1
                if old_sd=="B": bid_large-=1
                elif old_sd=="A": ask_large-=1
        large_nearby[i] = float(large_in_win > 0)
        iceberg_active[i] = float(o in iceberg_oids)
        tot = bid_large + ask_large
        large_imbalance[i] = (bid_large-ask_large)/(tot+1e-8) if tot>0 else 0.0
        ac = add_count + cancel_count
        cancel_rate[i] = cancel_count/ac if ac>0 else 0.0
    np.savez_compressed(str(out_path), large_nearby=large_nearby, iceberg_active=iceberg_active, large_imbalance=large_imbalance, cancel_rate=cancel_rate, n_events=np.array([n]), n_large=np.array([int(is_large.sum())]))
    log.info(f"{date_str}: {n} events, {int(is_large.sum())} large, {len(iceberg_oids)} icebergs saved")
    return True
def main():
    files = sorted(MBO_DIR.glob("*.mbo.dbn.zst"))
    log.info(f"Found {len(files)} MBO files")
    ok=0; fail=0
    for f in files:
        if process_day(f): ok+=1
        else: fail+=1
    log.info(f"DONE: {ok} ok, {fail} failed")
if __name__=="__main__": main()
