"""
evaluate_model.py
Loads a trained model and evaluates it, printing a confusion matrix and examples.
Usage:
python code/evaluate_model.py --data data/sample_dataset.csv --model models/best_model.pkl
"""
import argparse, joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data/sample_dataset.csv")
    p.add_argument("--model", default="../models/best_model.pkl")
    args = p.parse_args()
    df = pd.read_csv(args.data)
    model = joblib.load(args.model)
    X = df[["env_type","frequency_mhz","distance_m","num_walls","has_metal"]]
    y = df["best_ant"]
    preds = model.predict(X)
    print("Accuracy:", accuracy_score(y, preds))
    print("Confusion matrix:\\n", confusion_matrix(y, preds))
    # show some sample rows
    sample = df.sample(10, random_state=1)
    pred_s = model.predict(sample[["env_type","frequency_mhz","distance_m","num_walls","has_metal"]])
    print("\\nSample predictions (env,dist,num_walls,has_metal,freq) -> (pred) [actual]")
    for idx,row in sample.iterrows():
        print(row["env_type"], row["distance_m"], int(row["num_walls"]), int(row["has_metal"]), int(row["frequency_mhz"]),
              "->", int(pred_s[sample.index.get_loc(idx)]), "[", int(row["best_ant"]), "]")