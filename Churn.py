import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA_PATH = 'netflix_users.csv'
df = pd.read_csv(DATA_PATH)

CURRENT_DATE = datetime(2025, 4, 25)
df['Last_Login'] = pd.to_datetime(df['Last_Login'])
df['Churn'] = (CURRENT_DATE - df['Last_Login']).dt.days > 90
df['Churn'] = df['Churn'].astype(int)

plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.xlabel('Churn (1=Churned, 0=Active)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x='Subscription_Type', hue='Churn', data=df)
plt.title('Churn by Subscription Type')
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(df[df['Churn']==1]['Watch_Time_Hours'], color='red', label='Churned', kde=True, bins=30)
sns.histplot(df[df['Churn']==0]['Watch_Time_Hours'], color='green', label='Active', kde=True, bins=30)
plt.legend()
plt.title('Watch Time Distribution by Churn')
plt.xlabel('Watch Time (Hours)')
plt.show()

X = df.drop(['User_ID', 'Name', 'Last_Login', 'Churn'], axis=1)
y = df['Churn']

for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))

print("\nRecommendations to Reduce Churn:")
print("- Target users with low watch time and those who haven't logged in recently with re-engagement campaigns.")
print("- Offer special incentives to users on Basic/Standard plans who show signs of inactivity.")
print("- Analyze feedback from churned users to improve content and features.")
