import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_students = 200
data = {
    'Midterm_Grade': np.random.randint(0, 101, num_students),
    'Avg_Engagement_Score': np.random.randint(0, 101, num_students),
    'Assignments_Submitted': np.random.randint(0, 11, num_students), #out of 10
    'At_Risk': np.random.choice([0, 1], size=num_students, p=[0.8, 0.2]) # 20% at risk
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering ---
#No significant cleaning needed for synthetic data.  Feature Engineering could be added here.
#Example:  Creating a combined score.
df['Combined_Score'] = (df['Midterm_Grade'] + df['Avg_Engagement_Score'] + df['Assignments_Submitted']*10)/3
# --- 3. Data Splitting ---
X = df[['Midterm_Grade', 'Avg_Engagement_Score', 'Assignments_Submitted','Combined_Score']]
y = df['At_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 4. Model Training ---
model = LogisticRegression(solver='liblinear') # Choosing a solver appropriate for smaller datasets.
model.fit(X_train, y_train)
# --- 5. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
# --- 6. Visualization ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Midterm_Grade', y='Avg_Engagement_Score', hue='At_Risk', data=df, palette='viridis')
plt.title('Student Performance and Engagement')
plt.xlabel('Midterm Grade')
plt.ylabel('Average Engagement Score')
plt.savefig('student_performance.png')
print("Plot saved to student_performance.png")
plt.figure(figsize=(8,6))
sns.countplot(x='At_Risk', data=df)
plt.title('At Risk Student Count')
plt.xlabel('At Risk (0=No, 1=Yes)')
plt.ylabel('Count')
plt.savefig('at_risk_count.png')
print("Plot saved to at_risk_count.png")