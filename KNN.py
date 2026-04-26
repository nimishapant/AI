# Step 1: Import libraries
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report



# Load the locally saved Titanic CSV
df = pd.read_csv('/Users/cex/Desktop/AI/titanic.csv', sep=',')
# Preview the dataset
print(df.head())

# Step 3: Preprocess data
# Select features and target
features = ['pclass', 'sex', 'age', 'fare', 'embarked']
df = df[features + ['survived']]

# Drop rows with missing values
df = df.dropna()

# Encode categorical features
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df[features]
y = df['survived']

# Step 4: Split into training (60%), validation (20%), and testing (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
# Now: 60% train, 20% val, 20% test

# Step 5: Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 6: Find best K using validation data
accuracy_list = []
for k in range(1, 12):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_val_pred = knn.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_val_pred)
    accuracy_list.append(acc)
    print(f"K={k}, Validation Accuracy={acc:.4f}")

# Step 7: Train final model on train+val with best K
best_k = accuracy_list.index(max(accuracy_list)) + 1
print("\nBest K value based on validation set:", best_k)

# Combine train and validation sets

X_final_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train final model
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_final_train_scaled, y_train)


# Step 8: Evaluate on test set
y_test_pred = final_knn.predict(X_test_scaled)
print("\nFinal Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nFinal Classification Report:\n", classification_report(y_test, y_test_pred))