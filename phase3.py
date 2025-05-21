# natops_phase3.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports for preprocessing, modeling, and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import classification models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# 1. Data Loading
# Assuming the clustering results (features) and the target classes are stored in the file "natops_processed.csv".
# Adjust the path if necessary.
data_file = "natops_cluster_ratios.csv"
df = pd.read_csv(data_file)

# Split data into training and testing data
train_df = df[df["is_test"] == 0]
test_df = df[df["is_test"] == 1]

# Print basic summary of the data
print("Data shape:", df.shape)
print("Data preview:")
print(df.head())

# It is assumed that the dataset has one column representing the target action (e.g., 'action') 
# and other columns as features. Adjust the target column name accordingly.
# For example, assume that the target is stored in a column called "label".
target_column = "class"  # Change to the correct column name if necessary
if target_column not in df.columns:
    raise ValueError(f"Column '{target_column}' not found in the data. Please verify your data file.")

# 2. Data Preprocessing

# Separate features and labels
X_train = train_df.drop(columns=[target_column, 'sid', 'is_test'])
X_test = test_df.drop(columns=[target_column, 'sid', 'is_test'])
y_train = train_df[target_column]
y_test = test_df[target_column]

# It might be beneficial to examine correlations and perform feature selection,
# but for this template we proceed with all features.

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\nTrain/test split:")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 3. Model Training and Evaluation
# Define a helper function for model training and evaluation to avoid code duplication.
def train_and_evaluate(model, model_name="Model"):
    print(f"\n=== {model_name} ===")
    # Training
    model.fit(X_train_scaled, y_train)
    print("Training completed.")

    # Prediction on test set
    y_pred = model.predict(X_test_scaled)

    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    return model, acc

# --- Model 1: K-Nearest Neighbors ---
knn = KNeighborsClassifier(n_neighbors=5)
knn_model, knn_acc = train_and_evaluate(knn, "K-Nearest Neighbors")

# ---Multi-Layer Perceptron (MLP) Classifier ---
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_model, mlp_acc = train_and_evaluate(mlp, "MLP Classifier")

# 5. Performance Comparison

results = pd.DataFrame({
    "Model": ["KNN", "MLP Classifier"],
    "Test Accuracy": [knn_acc, mlp_acc]
})

print("\nModel Accuracies:")
print(results)
