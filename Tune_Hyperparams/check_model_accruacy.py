import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

data_path = "/home/ujx4ab/ondemand/WindTurbine_KL_Divergence/Data/combined_data.csv"
target_property = "gearbox_temp_bin"

filtered_df = pd.read_csv(data_path)
filtered_df.drop('timestamp', axis=1, inplace=True)

X = filtered_df.drop(target_property, axis=1)
y = filtered_df[target_property]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# initialize models w parameters
models = {
    "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=40, random_state=42),
    "Logistic Regression": LogisticRegression(C=0.5, solver='liblinear', penalty='l2', random_state=42),
    "Logistic Regression": LogisticRegression(C=1.0, solver='liblinear', penalty='l2', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=18)
}

for name, model in models.items():
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    print(f"{name} Results:")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}\n")