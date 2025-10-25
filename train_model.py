"""
train_model.py
Trains a RandomForest classifier to predict the best antenna (0/1/2) from features.
Saves model to models/best_model.pkl
Usage:
python code/train_model.py --data data/sample_dataset.csv --out models/best_model.pkl
"""
import argparse, joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(path):
    df = pd.read_csv(path)
    return df

def build_pipeline():
    cat = ["env_type", "frequency_mhz"]
    num = ["distance_m", "num_walls", "has_metal"]
    pre = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ], remainder="passthrough")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data/sample_dataset.csv")
    p.add_argument("--out", default="../models/best_model.pkl")
    args = p.parse_args()
    df = load_data(args.data)
    X = df[["env_type","frequency_mhz","distance_m","num_walls","has_metal"]]
    y = df["best_ant"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print("Confusion matrix:\\n", confusion_matrix(y_test, preds))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, outp)
    print("Saved model to", outp)