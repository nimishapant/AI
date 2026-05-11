import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load data
df = pd.read_csv("/Users/cex/Desktop/smsspamcollection-43213e63-5452-42f4-9233-27e2a444cd31/SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Vectorize messages
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Step 1: Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 2: 10-fold cross-validation on training set
model = MultinomialNB()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

all_true_cv = []
all_pred_cv = []

for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model.fit(X_tr, y_tr)
    y_val_pred = model.predict(X_val)

    all_true_cv.extend(y_val)
    all_pred_cv.extend(y_val_pred)

print("Cross-validation performance:\n")
print(classification_report(all_true_cv, all_pred_cv, target_names=["ham", "spam"]))

# Step 3: Train final model on full training set and evaluate on test set
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

print("Final test set performance:\n")
print(classification_report(y_test, y_test_pred, target_names=["ham", "spam"]))

# Plot confusion matrix for test set
conf_matrix = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["ham", "spam"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix on Test Set")
plt.grid(False)
plt.show()