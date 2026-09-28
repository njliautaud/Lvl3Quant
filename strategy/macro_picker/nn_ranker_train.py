"""
Neptune NN ranker — black-box ceiling (HC #561 R7e + HC #559 R6).

Architecture:
  - input: per-name per-date feature vector from feature_store/v1/* (regime EXCLUDED per HC #561 R2).
  - shallow 2-layer 4-head transformer over the date snapshot (set of N tickers
    -> N scalar scores), with positional encoding via the feature vector itself.
  - loss: pairwise BCE on sign of (fwd21_i - fwd21_j) for sampled pairs.
  - walk-forward: research/walk_forward.py defines fold boundaries;
    per fold we train -> score OOT -> compute long-top-decile minus short-bottom-decile
    daily portfolio metrics. Per-fold checkpoint saved.
  - MLflow logging mandatory.

Dispatch is via SSH wrapper on Neptune (Ray currently down — documented fallback).
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import mlflow

ROOT = Path(os.environ.get("LVL3_ROOT", "/home/nick/Lvl3Quant"))
FS = ROOT / "data/feature_store/v1"
CACHE = ROOT / "wheel_strategy_v1/data/cache"
OUT = ROOT / "strategy/macro_picker/nn_ranker_runs"
CKPT = OUT / "checkpoints"
LOG = ROOT / "logs/nn_ranker_train.log"

OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG, level=logging.INFO,
    format="%(asctime)s [nnrank] %(message)s",
)
log = logging.getLogger("nnrank")

# add path so we can import walk_forward
sys.path.insert(0, str(ROOT / "research"))
try:
    from walk_forward import (
        _annualize_sharpe, _annualize_sortino, _cagr, _max_dd, _calmar,
        _profit_factor, _win_rate, _verdict_for_metrics,
    )
except Exception:  # pragma: no cover - fallback inline metrics
    TRADING_DAYS = 252
    def _annualize_sharpe(r):
        r = r.dropna()
        if len(r) < 2 or r.std() == 0:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))
    def _annualize_sortino(r):
        r = r.dropna()
        d = r[r < 0]
        if len(d) < 1 or d.std() == 0:
            return float("nan")
        return float(r.mean() / d.std() * np.sqrt(TRADING_DAYS))
    def _cagr(r):
        r = r.dropna()
        if len(r) < 2: return float("nan")
        eq = (1.0 + r).cumprod()
        ny = len(r) / 252
        return float(eq.iloc[-1] ** (1.0 / ny) - 1.0) if ny > 0 else float("nan")
    def _max_dd(r):
        r = r.dropna()
        if len(r) < 2: return float("nan")
        eq = (1.0 + r).cumprod()
        return float(((eq - eq.cummax()) / eq.cummax()).min())
    def _calmar(r):
        c = _cagr(r); m = _max_dd(r)
        if not np.isfinite(c) or not np.isfinite(m) or m == 0: return float("nan")
        return float(c / abs(m))
    def _profit_factor(r):
        r = r.dropna()
        pos = r[r > 0].sum(); neg = -r[r < 0].sum()
        if neg == 0: return float("nan")
        return float(pos / neg)
    def _win_rate(r):
        r = r.dropna()
        return float((r > 0).mean()) if len(r) else float("nan")
    def _verdict_for_metrics(m):
        c = m.get("calmar", float("nan")); cagr = m.get("cagr", float("nan"))
        s = m.get("sharpe", float("nan")); md = m.get("max_dd", float("nan"))
        if not np.isfinite(c) or c < 1.0: return "FAILS CALMAR FLOOR"
        if np.isfinite(cagr) and cagr >= 0.25 and c >= 2.0: return "STRETCH MET"
        if (np.isfinite(cagr) and cagr >= 0.18 and c >= 1.5
            and np.isfinite(s) and s >= 1.5 and np.isfinite(md) and md >= -0.15):
            return "TARGET MET"
        return "VIABLE BUT UNDER-TARGET"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_panel(horizon: int = 21) -> tuple[pd.DataFrame, list[str]]:
    """Join feature families (NO regime), attach fwd21. Returns long panel + feature col list."""
    fund = pd.read_parquet(FS / "fund_features.parquet")
    flow = pd.read_parquet(FS / "flow_features.parquet")
    factor = pd.read_parquet(FS / "factor_features.parquet")
    theme = pd.read_parquet(FS / "theme_features.parquet")
    intraday = pd.read_parquet(FS / "intraday_features.parquet")

    df = fund.merge(flow, on=["ticker", "date"], how="outer")
    df = df.merge(factor, on=["ticker", "date"], how="outer")
    df = df.merge(theme, on=["ticker", "date"], how="outer")
    df = df.merge(intraday, on=["ticker", "date"], how="outer")
    drop_cols = [c for c in ["intraday_source", "sector_etf", "fund_asof"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    prices = pd.read_parquet(CACHE / "prices.parquet")[["ticker", "date", "close"]]
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["fwd_ret"] = prices.groupby("ticker")["close"].pct_change(horizon).shift(-horizon)
    df = df.merge(prices[["ticker", "date", "fwd_ret"]], on=["ticker", "date"], how="inner")
    df["date"] = pd.to_datetime(df["date"])

    feat_cols = [
        c for c in df.columns
        if c not in {"ticker", "date", "fwd_ret"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df, feat_cols


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class ShallowTransformerRanker(nn.Module):
    """Tokens = tickers. Each token's embedding = MLP(feature vector).
    Self-attention across the universe-of-the-day; output = scalar score per ticker."""

    def __init__(self, in_dim: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        # x: (B, N, F); mask: (B, N) — True means pad
        h = self.embed(x)
        h = self.transformer(h, src_key_padding_mask=mask)
        return self.head(h).squeeze(-1)  # (B, N)


class DateSnapshotDataset(Dataset):
    """One sample = one trading date. Returns (features, fwd_ret, mask)."""

    def __init__(self, panel: pd.DataFrame, feat_cols: list, dates: list,
                 ticker_index: dict[str, int]):
        self.panel = panel
        self.feat_cols = feat_cols
        self.dates = list(dates)
        self.ticker_index = ticker_index
        self.N = len(ticker_index)
        self.F = len(feat_cols)
        # pre-pivot: for each date, rows are tickers in fixed order, missing -> NaN.
        self._cache = {}

    def _build(self, date):
        if date in self._cache:
            return self._cache[date]
        sub = self.panel[self.panel["date"] == date]
        feats = np.full((self.N, self.F), np.nan, dtype=np.float32)
        fwd = np.full((self.N,), np.nan, dtype=np.float32)
        for _, row in sub.iterrows():
            i = self.ticker_index.get(row["ticker"])
            if i is None:
                continue
            feats[i] = [row[c] if pd.notna(row[c]) else np.nan for c in self.feat_cols]
            fwd[i] = row["fwd_ret"] if pd.notna(row["fwd_ret"]) else np.nan
        # mask: True if row is fully NaN or fwd missing => padding
        nan_row = np.all(np.isnan(feats), axis=1) | np.isnan(fwd)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        # normalize per-snapshot per-feature to z (robust)
        feats = _snapshot_normalize(feats, nan_row)
        self._cache[date] = (feats, fwd, nan_row)
        return self._cache[date]

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        feats, fwd, mask = self._build(self.dates[idx])
        return (
            torch.from_numpy(feats),
            torch.from_numpy(np.nan_to_num(fwd, nan=0.0)),
            torch.from_numpy(mask),
        )


def _snapshot_normalize(feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-feature z-score across valid rows; clipped to ±5."""
    valid = ~mask
    if valid.sum() < 2:
        return feats
    sub = feats[valid]
    mu = sub.mean(axis=0, keepdims=True)
    sd = sub.std(axis=0, keepdims=True) + 1e-6
    out = feats.copy()
    out[valid] = np.clip((sub - mu) / sd, -5.0, 5.0)
    return out


def pairwise_loss(scores: torch.Tensor, fwd: torch.Tensor, mask: torch.Tensor,
                  n_pairs: int = 256) -> torch.Tensor:
    """Sample pairs from non-masked tokens; BCE on sign of fwd diff."""
    B, N = scores.shape
    losses = []
    for b in range(B):
        valid = (~mask[b]).nonzero(as_tuple=True)[0]
        if len(valid) < 4:
            continue
        i = valid[torch.randint(0, len(valid), (n_pairs,), device=scores.device)]
        j = valid[torch.randint(0, len(valid), (n_pairs,), device=scores.device)]
        keep = i != j
        if keep.sum() == 0:
            continue
        i = i[keep]; j = j[keep]
        diff_score = scores[b, i] - scores[b, j]
        diff_fwd = fwd[b, i] - fwd[b, j]
        target = (diff_fwd > 0).float()
        losses.append(F.binary_cross_entropy_with_logits(diff_score, target))
    if not losses:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# walk-forward loop
# ---------------------------------------------------------------------------
@dataclass
class TrainCfg:
    horizon: int = 21
    train_months: int = 36
    oot_months: int = 12
    step_months: int = 6
    epochs: int = 8
    batch_size: int = 8
    lr: float = 3e-4
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    n_pairs: int = 256
    seed: int = 42


def _fold_dates(all_dates: list, cfg: TrainCfg) -> list[dict]:
    folds = []
    if not all_dates:
        return folds
    cursor = pd.Timestamp(all_dates[0])
    end = pd.Timestamp(all_dates[-1])
    while True:
        tr_start = cursor
        tr_end = tr_start + pd.DateOffset(months=cfg.train_months)
        oot_start = tr_end
        oot_end = oot_start + pd.DateOffset(months=cfg.oot_months)
        if oot_end > end + pd.Timedelta(days=1):
            break
        folds.append({
            "train_start": tr_start, "train_end": tr_end,
            "oot_start": oot_start, "oot_end": oot_end,
        })
        cursor = cursor + pd.DateOffset(months=cfg.step_months)
    return folds


def _portfolio_rets(score_df: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """Daily long-top-decile minus short-bottom-decile, h=21 forward => /21 daily."""
    df = panel[["ticker", "date", "fwd_ret"]].merge(
        score_df, on=["ticker", "date"], how="inner"
    ).dropna(subset=["score", "fwd_ret"])
    daily = []
    for date, grp in df.groupby("date"):
        if len(grp) < 10:
            continue
        q_lo = grp["score"].quantile(0.1)
        q_hi = grp["score"].quantile(0.9)
        longs = grp[grp["score"] >= q_hi]["fwd_ret"].mean()
        shorts = grp[grp["score"] <= q_lo]["fwd_ret"].mean()
        if np.isfinite(longs) and np.isfinite(shorts):
            daily.append((date, (longs - shorts) / 21.0))
    if not daily:
        return pd.Series(dtype=float)
    s = pd.Series(dict(daily)).sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def train_one_fold(panel: pd.DataFrame, feat_cols: list, fold: dict,
                   cfg: TrainCfg, device: str, ticker_index: dict, fold_idx: int) -> dict:
    torch.manual_seed(cfg.seed + fold_idx)
    np.random.seed(cfg.seed + fold_idx)

    train_dates = sorted(panel[(panel["date"] >= fold["train_start"]) &
                                (panel["date"] < fold["train_end"])]["date"].unique())
    oot_dates = sorted(panel[(panel["date"] >= fold["oot_start"]) &
                              (panel["date"] < fold["oot_end"])]["date"].unique())
    if len(train_dates) < 50 or len(oot_dates) < 20:
        return {"fold": fold_idx, "skipped": True, "reason": "insufficient_dates"}

    train_ds = DateSnapshotDataset(panel, feat_cols, train_dates, ticker_index)
    oot_ds = DateSnapshotDataset(panel, feat_cols, oot_dates, ticker_index)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=0, pin_memory=False)
    oot_dl = DataLoader(oot_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)

    model = ShallowTransformerRanker(
        in_dim=len(feat_cols), d_model=cfg.d_model,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers, dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for feats, fwd, mask in train_dl:
            feats = feats.to(device); fwd = fwd.to(device); mask = mask.to(device)
            scores = model(feats, mask=mask)
            loss = pairwise_loss(scores, fwd, mask, n_pairs=cfg.n_pairs)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        log.info(f"  fold {fold_idx} epoch {epoch} loss={mean_loss:.4f}")
        mlflow.log_metric(f"fold{fold_idx}_train_loss", mean_loss, step=epoch)

    # OOT scoring
    model.eval()
    rows = []
    inv_ticker = {v: k for k, v in ticker_index.items()}
    with torch.no_grad():
        for i, (feats, fwd, mask) in enumerate(oot_dl):
            feats = feats.to(device); mask = mask.to(device)
            scores = model(feats, mask=mask).cpu().numpy()
            # batch indices map to dates
            batch_dates = oot_dates[i * cfg.batch_size: i * cfg.batch_size + scores.shape[0]]
            for b_idx, d in enumerate(batch_dates):
                row_mask = mask[b_idx].cpu().numpy()
                for t_idx in range(scores.shape[1]):
                    if not row_mask[t_idx]:
                        rows.append({"ticker": inv_ticker[t_idx], "date": d,
                                     "score": float(scores[b_idx, t_idx])})
    score_df = pd.DataFrame(rows)

    rets = _portfolio_rets(score_df, panel[panel["date"].isin(oot_dates)])
    if rets.empty:
        return {"fold": fold_idx, "skipped": True, "reason": "no_rets"}

    metrics = {
        "sharpe": _annualize_sharpe(rets),
        "sortino": _annualize_sortino(rets),
        "cagr": _cagr(rets),
        "max_dd": _max_dd(rets),
        "calmar": _calmar(rets),
        "pf": _profit_factor(rets),
        "wr": _win_rate(rets),
    }
    metrics = {k: (None if not np.isfinite(v) else float(v)) for k, v in metrics.items()}

    ckpt_path = CKPT / f"fold_{fold_idx:02d}.pt"
    torch.save({"model": model.state_dict(),
                "feat_cols": feat_cols, "ticker_index": ticker_index,
                "cfg": cfg.__dict__, "metrics": metrics, "fold": fold}, ckpt_path)
    log.info(f"  fold {fold_idx} ckpt -> {ckpt_path}  metrics={metrics}")

    for k, v in metrics.items():
        if v is not None:
            mlflow.log_metric(f"fold{fold_idx}_{k}", v)

    return {"fold": fold_idx, **{k: v for k, v in fold.items()},
            **metrics, "ckpt": str(ckpt_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-folds", type=int, default=0, help="0 = all folds")
    ap.add_argument("--mlflow-uri", type=str,
                    default=os.environ.get("MLFLOW_URI", "http://jupiter:5000"))
    ap.add_argument("--experiment", type=str, default="macro_picker_nn_ranker")
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    cfg = TrainCfg(horizon=args.horizon, epochs=args.epochs,
                   batch_size=args.batch_size, lr=args.lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device: {device} | torch {torch.__version__}")
    if device == "cuda":
        log.info(f"gpu: {torch.cuda.get_device_name(0)} | "
                 f"mem total {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    run_name = args.run_name or f"nnrank_h{cfg.horizon}_{time.strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**cfg.__dict__, "device": device, "torch": torch.__version__})

        log.info("loading panel ...")
        panel, feat_cols = load_panel(horizon=cfg.horizon)
        log.info(f"panel: {panel.shape}  features: {len(feat_cols)}  "
                 f"tickers: {panel['ticker'].nunique()}  "
                 f"date range: {panel['date'].min()} -> {panel['date'].max()}")
        mlflow.log_params({
            "n_features": len(feat_cols),
            "n_tickers": int(panel["ticker"].nunique()),
            "n_rows": int(panel.shape[0]),
        })

        tickers = sorted(panel["ticker"].unique().tolist())
        ticker_index = {t: i for i, t in enumerate(tickers)}

        all_dates = sorted(panel["date"].unique())
        folds = _fold_dates(all_dates, cfg)
        log.info(f"folds: {len(folds)}")
        if args.max_folds > 0:
            folds = folds[: args.max_folds]
            log.info(f"limited to {len(folds)} folds")

        results = []
        for i, fold in enumerate(folds):
            log.info(f"=== fold {i}/{len(folds)-1}  "
                     f"train [{fold['train_start'].date()} .. {fold['train_end'].date()})  "
                     f"oot [{fold['oot_start'].date()} .. {fold['oot_end'].date()}) ===")
            try:
                res = train_one_fold(panel, feat_cols, fold, cfg, device,
                                     ticker_index, i)
                results.append(res)
                with open(OUT / "results_running.json", "w") as f:
                    json.dump(results, f, indent=2, default=str)
            except Exception as e:
                log.exception(f"fold {i} failed: {e}")
                results.append({"fold": i, "error": str(e)})

        # aggregate
        ok = [r for r in results if "error" not in r and not r.get("skipped")]
        agg = {}
        if ok:
            for k in ("sharpe", "sortino", "cagr", "max_dd", "calmar", "pf", "wr"):
                vals = [r[k] for r in ok if r.get(k) is not None]
                if vals:
                    s = pd.Series(vals)
                    agg[k] = {"median": float(s.median()),
                              "p25": float(s.quantile(0.25)),
                              "p75": float(s.quantile(0.75))}
        med = {k: v["median"] for k, v in agg.items()}
        verdict = _verdict_for_metrics(med) if med else "FAILS CALMAR FLOOR"
        log.info(f"verdict: {verdict}  agg medians: {med}")
        mlflow.log_metric("verdict_code", {"FAILS CALMAR FLOOR": 0, "VIABLE BUT UNDER-TARGET": 1,
                                            "TARGET MET": 2, "STRETCH MET": 3}.get(verdict, -1))
        for k, v in med.items():
            mlflow.log_metric(f"agg_median_{k}", v)

        with open(OUT / "results_final.json", "w") as f:
            json.dump({"folds": results, "summary": agg, "verdict": verdict,
                       "config": cfg.__dict__}, f, indent=2, default=str)
        mlflow.log_artifact(str(OUT / "results_final.json"))


if __name__ == "__main__":
    main()
