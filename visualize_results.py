import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Paths
designs_path = os.path.join("designs")
model_path = os.path.join("models", "best_model.pkl")
dataset_path = os.path.join("data", "antenna_dataset.csv")  # replace with your CSV name

# 1. Show fractal antenna designs (PNG)
def plot_antenna_designs():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = ["sierpinski.png", "koch.png", "monopole.png"]
    for i, name in enumerate(names):
        img_path = os.path.join(designs_path, name)
        img = mpimg.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(name.split(".")[0].capitalize())
        axes[i].axis("off")
    plt.suptitle("Fractal Antenna Designs", fontsize=14)
    plt.show()

# 2. Feature distribution plots
def plot_feature_distributions():
    df = pd.read_csv(dataset_path)
    features = df.columns[:-1]  # assume last column is label
    for feature in features:
        plt.figure(figsize=(6,4))
        sns.boxplot(x="label", y=feature, data=df)
        plt.title(f"Distribution of {feature} across antenna classes")
        plt.show()

# 3. Confusion matrix visualization
def plot_confusion_matrix():
    df = pd.read_csv(dataset_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    model = joblib.load(model_path)
    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Sierpinski (0)", "Koch (1)", "Monopole (2)"]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix Heatmap")
    plt.show()

if __name__ == "__main__":
    plot_antenna_designs()
    plot_feature_distributions()
    plot_confusion_matrix()
