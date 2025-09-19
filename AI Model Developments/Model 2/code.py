import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Create target 'passed'
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['passed'] = (df['average_score'] >= 60).astype(int)

# Separate features and target
X = df.drop(columns=['average_score', 'passed'])
y = df['passed']

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict from user input
print("\n🔮 Predict Student Result")

# Ask user for each feature manually
gender = input("Enter gender (male/female): ").lower()
race = input("Enter race/ethnicity (group A/B/C/D/E): ").title()
parent = input("Enter parental education (high school/some college/bachelor's/master's etc.): ").lower()
lunch = input("Enter lunch type (standard/free/reduced): ").lower()
course = input("Test preparation course (none/completed): ").lower()
math = float(input("Enter Math score: "))
reading = float(input("Enter Reading score: "))
writing = float(input("Enter Writing score: "))

# Create a dataframe with same columns as X
user_df = pd.DataFrame([{
    'gender': gender,
    'race/ethnicity': race,
    'parental level of education': parent,
    'lunch': lunch,
    'test preparation course': course,
    'math score': math,
    'reading score': reading,
    'writing score': writing
}])

# One-hot encode user input using same columns as training set
user_df = pd.get_dummies(user_df)
user_df = user_df.reindex(columns=X.columns, fill_value=0)

# Predict
prediction = model.predict(user_df)
if prediction[0] == 1:
    print("✅ The student is likely to PASS")
else:
    print("❌ The student is likely to FAIL")
