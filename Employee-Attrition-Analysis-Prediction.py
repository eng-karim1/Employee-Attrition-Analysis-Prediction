import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load data
data_path = '/mnt/data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
df = pd.read_csv(data_path)

# Overview of the data
display(df.info())
display(df.head())

# Check for missing values
display(df.isnull().sum())

# Visualizing attrition distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Attrition', data=df, palette='pastel')
plt.title('Employee Attrition Distribution')
plt.xlabel('Attrition Status')
plt.ylabel('Count')
plt.show()

# Encode target variable
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

# Drop irrelevant columns
df.drop(columns=['EmployeeNumber', 'Over18', 'EmployeeCount', 'StandardHours'], inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split data into training and testing sets
X = df.drop(columns=['Attrition'])
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model using XGBoost
model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6, 4))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Attrition', 'Attrition'], yticklabels=['No Attrition', 'Attrition'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance analysis
feature_importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
plt.figure(figsize=(8, 5))
feature_importances.sort_values().plot(kind='barh', title='Top Features Influencing Employee Attrition', color='teal')
plt.xlabel('Importance Score')
plt.show()

# Deeper Analysis: Understanding Attrition Factors
plt.figure(figsize=(8, 5))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, palette='coolwarm')
plt.title('Monthly Income vs Attrition')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Attrition', y='JobSatisfaction', data=df, palette='coolwarm')
plt.title('Job Satisfaction vs Attrition')
plt.show()

