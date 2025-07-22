import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# --- Load and clean dataset ---
df = pd.read_csv('dataset.csv')
df_clean = df.dropna()

# Strip quotes and whitespace from duration columns
df_clean['Sleep Duration'] = df_clean['Sleep Duration'].astype(str).str.strip().str.replace('"', '').str.replace("'", '')
df_clean['Work/Study Hours'] = df_clean['Work/Study Hours'].astype(str).str.strip().str.replace('"', '').str.replace("'", '')

# Replace string durations with numeric values
df_clean['Sleep Duration'] = df_clean['Sleep Duration'].replace({
    'Less than 5 hours': 4,
    '5-6 hours': 5.5,
    '6-7 hours': 6.5,
    '7-8 hours': 7.5,
    'More than 8 hours': 9
})

df_clean['Work/Study Hours'] = df_clean['Work/Study Hours'].replace({
    'Less than 2': 1,
    '2-4': 3,
    '4-6': 5,
    'More than 6': 7
})

# Convert to numeric and drop invalid rows
df_clean['Sleep Duration'] = pd.to_numeric(df_clean['Sleep Duration'], errors='coerce')
df_clean['Work/Study Hours'] = pd.to_numeric(df_clean['Work/Study Hours'], errors='coerce')
df_clean = df_clean.dropna(subset=['Sleep Duration', 'Work/Study Hours'])

# --- Label Encode Categorical Features ---
categorical_cols = [
    'Gender', 'City', 'Profession',
    'Dietary Habits', 'Degree',
    'Have you ever had suicidal thoughts ?',
    'Family History of Mental Illness', 'Financial Stress'
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# --- Select Features and Target ---
feature_cols = [
     'Age', 'Academic Pressure',
    'CGPA', 'Sleep Duration', 'Have you ever had suicidal thoughts ?',
    'Work/Study Hours'
    #, 'Financial Stress'
]

X = df_clean[feature_cols]
y = df_clean['Depression']

# --- Scale Numerical Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- Train Logistic Regression Model ---
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Save Model & Preprocessing Objects ---
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoder.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# --- Evaluation ---
y_pred = model.predict(X_test)
print("\n✅ Model Trained Successfully")
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
