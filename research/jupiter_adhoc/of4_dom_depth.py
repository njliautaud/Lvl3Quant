import numpy as np, json, logging
from pathlib import Path
logging.basicConfig(format="%(asctime)s [of4] %(message)s", level=logging.INFO)
log = logging.getLogger("of4")
BOOK_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot")
OF1_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/orderflow_features")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth")
OUT_DIR.mkdir(parents=True, exist_ok=True)
def process_day(book_path):
    date_str = book_path.stem.replace("_book_tensors","").replace("-","")
    out_path = OUT_DIR / f"{date_str}_of4.npz"
    if out_path.exists(): log.info(f"skip {date_str}"); return True
    d = np.load(str(book_path))
    bt = d["book_tensors"]
    mid = d["mid_prices"].astype(np.float32)
    bid_sz = bt[:, :10, 1]
    ask_sz = bt[:, 10:, 2]
    bid5 = bid_sz[:,:5].sum(axis=1).astype(np.float32)
    ask5 = ask_sz[:,:5].sum(axis=1).astype(np.float32)
    bid10 = bid_sz.sum(axis=1).astype(np.float32)
    ask10 = ask_sz.sum(axis=1).astype(np.float32)
    imb5 = (bid5-ask5)/(bid5+ask5+1e-8)
    imb10 = (bid10-ask10)/(bid10+ask10+1e-8)
    bid_wall = bid_sz.max(axis=1).astype(np.float32)
    ask_wall = ask_sz.max(axis=1).astype(np.float32)
    wall_imb = (bid_wall-ask_wall)/(bid_wall+ask_wall+1e-8)
    poc_dist = np.zeros(len(mid), dtype=np.float32)
    of1_path = OF1_DIR / f"{date_str}_orderflow_summary.json"
    if of1_path.exists():
        s = json.loads(of1_path.read_text())
        poc = s.get("session_poc",0)
        if poc and poc > 0: poc_dist = ((mid - poc) / 0.25).astype(np.float32)
    np.savez_compressed(str(out_path), depth_imbalance_5=imb5, depth_imbalance_10=imb10, bid_depth_5=bid5, ask_depth_5=ask5, bid_wall=bid_wall, ask_wall=ask_wall, wall_imbalance=wall_imb, poc_dist_ticks=poc_dist)
    log.info(f"{date_str}: {len(mid)} bars, mean_imb5={imb5.mean():.3f} saved")
    return True
def main():
    files = sorted(BOOK_DIR.glob("*_book_tensors.npz"))
    log.info(f"Found {len(files)} book cache files")
    ok=0; fail=0
    for f in files:
        if process_day(f): ok+=1
        else: fail+=1
    log.info(f"DONE: {ok} ok, {fail} failed")
if __name__=="__main__": main()
