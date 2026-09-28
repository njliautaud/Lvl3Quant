#!/usr/bin/env python3
"""
research_dashboard.py — Quick summary of all quant research status.
Run: python3 scripts/research_dashboard.py
"""
import json, os, glob
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")

def count_kb_findings():
    """Count findings in knowledge base"""
    kb = ROOT / "research/QUANT_KNOWLEDGE_BASE.md"
    if not kb.exists():
        return 0
    count = 0
    with open(kb) as f:
        for line in f:
            if line.strip().startswith("### Finding #"):
                count += 1
    return count

def count_experiments():
    """Count MLflow experiments"""
    try:
        import subprocess
        r = subprocess.run(["python3", "-c", 
            "import mlflow; mlflow.set_tracking_uri('http://localhost:5000'); print(len(mlflow.search_experiments()))"],
            capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip()) if r.returncode == 0 else "?"
    except:
        return "?"

def count_paper_engines():
    """Count paper engine scripts"""
    engines = list((ROOT / "paper_engines").glob("*_paper.py"))
    return len(engines)

def get_paper_engine_states():
    """Get status of paper engines with state files"""
    states = {}
    for sf in sorted((ROOT / "state").glob("*paper*state*.json")):
        try:
            with open(sf) as f:
                d = json.load(f)
            name = sf.stem.replace("_state", "").replace("_paper", "")
            positions = len(d.get("open_positions", d.get("positions", [])))
            equity = d.get("equity", d.get("capital", "?"))
            states[name] = {"positions": positions, "equity": equity}
        except:
            pass
    return states

def get_best_strategies():
    """Extract best validated strategies from results"""
    results = []
    # Check for results JSON files
    for rj in (ROOT / "output/growth_research").rglob("*results*.json"):
        try:
            with open(rj) as f:
                d = json.load(f)
            if isinstance(d, dict) and "sharpe" in str(d).lower():
                name = rj.parent.name
                # Try to extract Sharpe
                if "baseline" in d and isinstance(d["baseline"], dict):
                    sh = d["baseline"].get("sharpe", None)
                elif "sharpe" in d:
                    sh = d["sharpe"]
                else:
                    sh = None
                if sh and isinstance(sh, (int, float)) and sh > 1.0:
                    results.append((name, sh))
        except:
            pass
    return sorted(results, key=lambda x: -x[1])[:10]

def main():
    print("=" * 70)
    print(f"  QUANT RESEARCH DASHBOARD — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    print(f"\n📊 Knowledge Base: {count_kb_findings()} findings")
    print(f"🧪 MLflow Experiments: {count_experiments()}")
    print(f"📈 Paper Engines: {count_paper_engines()}")
    
    # Research scripts
    scripts = list((ROOT / "scripts/growth_research").glob("*.py"))
    print(f"📝 Research Scripts: {len(scripts)}")
    
    # Paper engine states
    states = get_paper_engine_states()
    if states:
        print(f"\n--- Active Paper Engines ({len(states)} with state) ---")
        for name, info in sorted(states.items()):
            print(f"  {name}: {info['positions']} positions, equity ${info['equity']}")
    
    # Best strategies
    best = get_best_strategies()
    if best:
        print(f"\n--- Top Validated Strategies (by Sharpe) ---")
        for name, sh in best[:8]:
            print(f"  {name}: Sharpe {sh:.2f}")
    
    # V8 config
    print(f"\n--- Production Config (V8) ---")
    print(f"  DTE=14 | OTM=2% | Weekly rebal | 17 features | LGBM")
    print(f"  Backtest: Sharpe 3.23, 5/5 gates, MDD -8.0%")
    print(f"  Monte Carlo CI: [2.39, 3.64] (100% positive)")
    print(f"  Structural edge: ~90% (robust to ML decay)")
    
    # Portfolio combination
    print(f"\n--- Optimal Portfolio Combination ---")
    print(f"  Regime-switched: Sharpe 4.35, MaxDD -4.9%, CAGR 26.9%")
    print(f"  Weights: CTA 34%, ETF Rot 26%, CTA+Sector 23%, Sector ML 12%")
    
    print(f"\n{'=' * 70}")

if __name__ == "__main__":
    main()
