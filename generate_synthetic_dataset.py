"""
generate_synthetic_dataset.py
Generates a synthetic RSSI dataset in CSV format for the ML pipeline.
Usage:
python code/generate_synthetic_dataset.py --out data/sample_dataset.csv --n_samples 2000
"""
import argparse, csv
from pathlib import Path
import numpy as np

def generate_synthetic(n_samples=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    env_types = ["indoor","urban","rural"]
    freqs = [915, 2400]  # MHz
    rows = []
    offsets = {
        "indoor": [0.5, 0.0, -1.0],
        "urban": [0.0, 0.7, -0.5],
        "rural": [-0.5, -0.2, 1.0]
    }
    for _ in range(n_samples):
        env = rng.choice(env_types, p=[0.4,0.4,0.2])
        d = rng.uniform(1,200)  # meters
        num_walls = int(rng.poisson(1.2)) if env=="indoor" else 0
        has_metal = int(rng.choice([0,1], p=[0.85,0.15]))
        f = float(rng.choice(freqs))
        Pt = 0.0
        d0 = 1.0
        n = {"indoor":3.0, "urban":2.7, "rural":2.0}[env] * (1.0 + (2400-f)/2400*0.05)
        PLd0 = 30 + (2400-f)/1000.0
        base_rssi = Pt - PLd0 - 10*n*np.log10(max(d,d0)/d0)
        base_rssi -= num_walls*3.5
        if has_metal:
            base_rssi -= 6.0
        ant_rssis = []
        for ai in range(3):
            ant_offset = offsets[env][ai]
            freq_pen = 0.0
            if ai==2 and f==915:
                freq_pen += 0.6
            if ai==0 and f==2400:
                freq_pen += 0.2
            noise = rng.normal(0,2.5)
            rssi = base_rssi + ant_offset + freq_pen + noise
            ant_rssis.append(round(float(rssi),3))
        best = int(np.argmax(ant_rssis))
        rows.append({
            "env_type":env,
            "distance_m":round(float(d),3),
            "num_walls":int(num_walls),
            "has_metal":int(has_metal),
            "frequency_mhz":float(f),
            "ant_0_rssi_dbm":ant_rssis[0],
            "ant_1_rssi_dbm":ant_rssis[1],
            "ant_2_rssi_dbm":ant_rssis[2],
            "best_ant":best
        })
    return rows

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="../data/sample_dataset.csv")
    p.add_argument("--n_samples", type=int, default=2000)
    args = p.parse_args()
    rows = generate_synthetic(args.n_samples)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(outp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print("Wrote", outp)