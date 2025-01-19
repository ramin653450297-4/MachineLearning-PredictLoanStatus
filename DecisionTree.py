import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('loan.csv')

# Encode categorical variables
data['gender'] = data['gender'].map({'Female': 0, 'Male': 1})
data = pd.get_dummies(data, columns=['occupation'], prefix='occupation')
data['education_level'] = data['education_level'].map({
    'Bachelor': 0, 'Master': 1, 'High School': 2, 'Associate': 3, 'Doctoral': 4
})
data['marital_status'] = data['marital_status'].map({
    'Single': 0, 'Married': 1
})
data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Denied': 0})

# Handle missing values if any
data = data.dropna()

# Features and target
feature_columns = ['age', 'gender','education_level'] + [col for col in data.columns if col.startswith('occupation_')] + ['marital_status', 'income'] 
X = data[feature_columns]
y = data['loan_status']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree model
dt_model = DecisionTreeClassifier()

# Train the model
dt_model.fit(X_train, y_train)

# Predict
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
error_dt = (1 - accuracy_dt) * 100
print(f"Decision Tree Accuracy = {accuracy_dt * 100:.2f}%")
print(f"Decision Tree Error = {error_dt:.2f}%")

# Save the model
joblib.dump(dt_model, 'Loan_decision_tree.pkl')
print("Model saved as Loan_decision_tree.pkl")
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")
joblib.dump(feature_columns, 'feature_columns.pkl')
print(f"Feature columns saved as feature_columns.pkl: {len(feature_columns)} features")

