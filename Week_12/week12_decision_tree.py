
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

OUT = "plots/"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("Iris.csv")
df = df.drop(columns="Id")
df["Species"] = df["Species"].str.replace("Iris-", "", regex=False)
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

X, y = df[features], df["Species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("=========== DECISION TREE RESULTS ===========")
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Accuracy            : {accuracy_score(y_test, pred):.4f}")
print(f"Precision (macro)   : {precision_score(y_test, pred, average='macro'):.4f}")
print(f"Recall (macro)      : {recall_score(y_test, pred, average='macro'):.4f}")
print(f"Precision (weighted): {precision_score(y_test, pred, average='weighted'):.4f}")
print(f"Recall (weighted)   : {recall_score(y_test, pred, average='weighted'):.4f}")
print("\nClassification report:\n", classification_report(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("\nFeature importances:")
for f, i in sorted(zip(features, model.feature_importances_), key=lambda t: -t[1]):
    print(f"  {f:15s} {i:.4f}")

ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues")
plt.title("Confusion Matrix - Decision Tree"); plt.tight_layout()
plt.savefig(OUT + "confusion_matrix.png", dpi=150); plt.close()

plt.figure(figsize=(14, 8))
plot_tree(model, feature_names=features, class_names=model.classes_,
          filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree"); plt.tight_layout()
plt.savefig(OUT + "decision_tree.png", dpi=150); plt.close()

print("\nPlots saved in the 'plots' folder.")
