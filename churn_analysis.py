import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("customer_churn.csv")

# Encode categorical columns
le_contract = LabelEncoder()
le_internet = LabelEncoder()
le_churn = LabelEncoder()

df["ContractType"] = le_contract.fit_transform(df["ContractType"])
df["InternetService"] = le_internet.fit_transform(df["InternetService"])
df["Churn"] = le_churn.fit_transform(df["Churn"])

# Features and target
X = df[["Tenure", "MonthlyCharges", "ContractType", "InternetService"]]
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
